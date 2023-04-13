import os
import time

import torch
import torch.multiprocessing
from torch.nn.utils.rnn import pad_sequence
from torch.optim import RAdam
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from TrainingInterfaces.Text_to_Spectrogram.AutoAligner.Aligner import Aligner
from TrainingInterfaces.Text_to_Spectrogram.AutoAligner.TinyTTS import TinyTTS


def collate_and_pad(batch):
    # text, text_len, speech, speech_len
    return (pad_sequence([datapoint[0] for datapoint in batch], batch_first=True),
            torch.stack([datapoint[1] for datapoint in batch]).squeeze(1),
            pad_sequence([datapoint[2] for datapoint in batch], batch_first=True),
            torch.stack([datapoint[3] for datapoint in batch]).squeeze(1),
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
               use_reconstruction=True):
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
        use_reconstruction: whether to use the auxiliary spectrogram reconstruction procedure/loss, which can make the alignment sharper
    """
    os.makedirs(save_directory, exist_ok=True)
    torch.multiprocessing.set_sharing_strategy('file_system')
    train_loader = DataLoader(batch_size=batch_size,
                              dataset=train_dataset,
                              drop_last=True,
                              num_workers=8 if os.cpu_count() > 8 else max(os.cpu_count() - 2, 1),
                              pin_memory=False,
                              shuffle=True,
                              prefetch_factor=16,
                              collate_fn=collate_and_pad,
                              persistent_workers=True)

    asr_model = Aligner().to(device)
    optim_asr = RAdam(asr_model.parameters(), lr=0.0001)

    tiny_tts = TinyTTS().to(device)
    optim_tts = RAdam(tiny_tts.parameters(), lr=0.0001)

    step_counter = 0
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
        loss_sum = list()

        asr_model.train()
        tiny_tts.train()
        for batch in tqdm(train_loader):
            tokens = batch[0].to(device)
            tokens_len = batch[1].to(device)
            mel = batch[2].to(device)
            mel_len = batch[3].to(device)
            speaker_embeddings = batch[4].to(device)

            pred = asr_model(mel, mel_len)

            ctc_loss = asr_model.ctc_loss(pred.transpose(0, 1).log_softmax(2),
                                          tokens,
                                          mel_len,
                                          tokens_len)

            if use_reconstruction:
                speaker_embeddings_expanded = torch.nn.functional.normalize(speaker_embeddings).unsqueeze(1).expand(-1, pred.size(1), -1)
                tts_lambda = min([5, step_counter / 2000])  # super simple schedule
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

            step_counter += 1

            loss_sum.append(loss.item())

        asr_model.eval()
        loss_this_epoch = sum(loss_sum) / len(loss_sum)
        torch.save({
            "asr_model"    : asr_model.state_dict(),
            "optimizer"    : optim_asr.state_dict(),
            "tts_model"    : tiny_tts.state_dict(),
            "tts_optimizer": optim_tts.state_dict(),
            "step_counter" : step_counter,
        },
            os.path.join(save_directory, "aligner.pt"))
        print("Total Loss:   {}".format(round(loss_this_epoch, 3)))
        print("Time elapsed: {} Minutes".format(round((time.time() - start_time) / 60)))
        print("Steps:        {}".format(step_counter))
        if debug_img_path is not None:
            asr_model.inference(mel=mel[0][:mel_len[0]],
                                tokens=tokens[0][:tokens_len[0]],
                                save_img_for_debug=debug_img_path + f"/{step_counter}.png",
                                train=True)  # for testing
        if step_counter > steps:
            return
