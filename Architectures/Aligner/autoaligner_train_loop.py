import os
import time

import torch
import torch.multiprocessing
from torch.nn.utils.rnn import pad_sequence
from torch.optim import RAdam
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from Architectures.Aligner.Aligner import Aligner
from Architectures.Aligner.Reconstructor import Reconstructor
from Preprocessing.AudioPreprocessor import AudioPreprocessor
from Preprocessing.EnCodecAudioPreprocessor import CodecAudioPreprocessor


def collate_and_pad(batch):
    # text, text_len, speech, speech_len, embed
    return (pad_sequence([datapoint[0] for datapoint in batch], batch_first=True),
            torch.stack([datapoint[1] for datapoint in batch]).squeeze(1),
            [datapoint[2] for datapoint in batch],
            None,
            torch.stack([datapoint[4] for datapoint in batch]).squeeze())


def train_loop(train_dataset,
               device,
               save_directory,
               batch_size,
               steps,
               path_to_checkpoint=None,
               fine_tune=False,
               resume=False,
               debug_img_path=None,
               use_reconstruction=True,
               gpu_count=1,
               rank=0,
               steps_per_checkpoint=None):
    """
    Args:
        resume: whether to resume from the most recent checkpoint
        steps: How many steps to train
        path_to_checkpoint: reloads a checkpoint to continue training from there
        fine_tune: whether to load everything from a checkpoint, or only the model parameters
        train_dataset: Pytorch Dataset Object for train data
        device: Device to put the loaded tensors on
        save_directory: Where to save the checkpoints
        batch_size: How many elements should be loaded at once
        debug_img_path: where to put images of the training progress if desired
        use_reconstruction: whether to use the auxiliary reconstruction procedure/loss, which can make the alignment sharper
    """
    os.makedirs(save_directory, exist_ok=True)
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.set_start_method('spawn', force=True)

    if steps_per_checkpoint is None:
        steps_per_checkpoint = len(train_dataset) // batch_size
    ap = CodecAudioPreprocessor(input_sr=-1, device=device)  # only used to transform features into continuous matrices
    spectrogram_extractor = AudioPreprocessor(input_sr=16000, output_sr=16000, device=device)

    asr_model = Aligner().to(device)
    optim_asr = RAdam(asr_model.parameters(), lr=0.0001)

    tiny_tts = Reconstructor().to(device)
    optim_tts = RAdam(tiny_tts.parameters(), lr=0.0001)

    if gpu_count > 1:
        asr_model.to(rank)
        tiny_tts.to(rank)
        asr_model = torch.nn.parallel.DistributedDataParallel(
            asr_model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True,
        ).module
        tiny_tts = torch.nn.parallel.DistributedDataParallel(
            tiny_tts,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True,
        ).module
        torch.distributed.barrier()
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    batch_sampler_train = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)

    train_loader = DataLoader(dataset=train_dataset,
                              num_workers=0,  # unfortunately necessary for big data due to mmap errors
                              batch_sampler=batch_sampler_train,
                              prefetch_factor=None,
                              collate_fn=collate_and_pad)

    step_counter = 0
    loss_sum = list()

    if resume:
        previous_checkpoint = os.path.join(save_directory, "aligner.pt")
        path_to_checkpoint = previous_checkpoint
        fine_tune = False

    if path_to_checkpoint is not None:
        check_dict = torch.load(os.path.join(path_to_checkpoint), map_location=device)
        asr_model.load_state_dict(check_dict["asr_model"])
        tiny_tts.load_state_dict(check_dict["tts_model"])
        if not fine_tune:
            optim_asr.load_state_dict(check_dict["optimizer"])
            optim_tts.load_state_dict(check_dict["tts_optimizer"])
            step_counter = check_dict["step_counter"]
            if step_counter > steps:
                print("Desired steps already reached in loaded checkpoint.")
                return
    start_time = time.time()

    while True:
        asr_model.train()
        tiny_tts.train()
        for batch in tqdm(train_loader):
            tokens = batch[0].to(device)
            tokens_len = batch[1].to(device)
            speaker_embeddings = batch[4].to(device)

            mels = list()
            mel_lengths = list()
            for datapoint in batch[2]:
                with torch.inference_mode():
                    # extremely unfortunate that we have to do this over here, but multiprocessing and this don't go together well
                    speech = ap.indexes_to_audio(datapoint.int().to(device))
                    mel = spectrogram_extractor.audio_to_mel_spec_tensor(speech, explicit_sampling_rate=16000).transpose(0, 1).cpu()
                speech_len = torch.LongTensor([len(mel)])
                mels.append(mel.clone())
                mel_lengths.append(speech_len)
            mel = pad_sequence(mels, batch_first=True).to(device)
            mel_len = torch.stack(mel_lengths).squeeze(1).to(device)

            pred = asr_model(mel, mel_len)

            ctc_loss = asr_model.ctc_loss(pred.transpose(0, 1).log_softmax(2),
                                          tokens,
                                          mel_len,
                                          tokens_len)

            if use_reconstruction:
                speaker_embeddings_expanded = torch.nn.functional.normalize(speaker_embeddings).unsqueeze(1).expand(-1, pred.size(1), -1)
                tts_lambda = min([0.1, step_counter / 10000])  # super simple schedule
                reconstruction_loss = tiny_tts(x=torch.cat([pred, speaker_embeddings_expanded], dim=-1),
                                               # combine ASR prediction with speaker embeddings to allow for reconstruction loss on multiple speakers
                                               lens=mel_len,
                                               ys=mel) * tts_lambda  # reconstruction loss to make the states more distinct
                loss = ctc_loss + reconstruction_loss
            else:
                loss = ctc_loss

            optim_asr.zero_grad()
            if use_reconstruction:
                optim_tts.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(asr_model.parameters(), 1.0)
            if use_reconstruction:
                torch.nn.utils.clip_grad_norm_(tiny_tts.parameters(), 1.0)
            optim_asr.step()
            if use_reconstruction:
                optim_tts.step()

            loss_sum.append(loss.item())
            step_counter += 1

            if step_counter % steps_per_checkpoint == 0 and rank == 0:
                asr_model.eval()
                torch.save({
                    "asr_model"    : asr_model.state_dict(),
                    "optimizer"    : optim_asr.state_dict(),
                    "tts_model"    : tiny_tts.state_dict(),
                    "tts_optimizer": optim_tts.state_dict(),
                    "step_counter" : step_counter,
                },
                    os.path.join(save_directory, "aligner.pt"))
                print("Total Loss:   {}".format(round(sum(loss_sum) / len(loss_sum), 3)))
                print("Time elapsed: {} Minutes".format(round((time.time() - start_time) / 60)))
                print("Steps:        {}".format(step_counter))
                if debug_img_path is not None:
                    asr_model.inference(features=mel[0][:mel_len[0]],
                                        tokens=tokens[0][:tokens_len[0]],
                                        save_img_for_debug=debug_img_path + f"/{step_counter}.png",
                                        train=True)  # for testing
                asr_model.train()
                loss_sum = list()

        if step_counter > steps and step_counter % steps_per_checkpoint == 0:
            return
