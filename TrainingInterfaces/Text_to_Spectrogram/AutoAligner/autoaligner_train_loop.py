import os
import time

import torch
import torch.multiprocessing
import torch.nn.functional as F
from torch.nn import CTCLoss
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader

from TrainingInterfaces.Text_to_Spectrogram.AutoAligner.Aligner import Aligner, TTSModel
from Utility.utils import delete_old_checkpoints
from Utility.utils import get_most_recent_checkpoint


def collate_and_pad(batch):
    # text, text_len, speech, speech_len
    return (pad_sequence([datapoint[0] for datapoint in batch], batch_first=True),
            torch.stack([datapoint[1] for datapoint in batch]).squeeze(1),
            pad_sequence([datapoint[2] for datapoint in batch], batch_first=True),
            torch.stack([datapoint[3] for datapoint in batch]).squeeze(1))


def train_loop(train_dataset,
               device,
               save_directory,
               batch_size,
               steps,
               epochs_per_save,
               path_to_checkpoint=None,
               fine_tune=False,
               resume=False):
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
        epochs_per_save: how many epochs to train in between checkpoints
    """
    train_loader = DataLoader(batch_size=batch_size,
                              dataset=train_dataset,
                              drop_last=True,
                              num_workers=8,
                              pin_memory=False,
                              shuffle=True,
                              prefetch_factor=16,
                              collate_fn=collate_and_pad,
                              persistent_workers=True)

    asr_model = Aligner(n_mels=80,
                        num_symbols=144)
    tts_model = TTSModel(n_mels=80,
                         num_symbols=+ 1)
    optim_asr = Adam(asr_model.parameters(), lr=1e-4)
    optim_tts = Adam(tts_model.parameters(), lr=1e-4)

    ctc_loss = CTCLoss()

    step_counter = 0
    epoch = 0
    if resume:
        previous_checkpoint = get_most_recent_checkpoint(checkpoint_dir=save_directory)
        if previous_checkpoint is not None:
            path_to_checkpoint = previous_checkpoint
            fine_tune = False
        else:
            fine_tune = True

    if path_to_checkpoint is not None:
        check_dict = torch.load(os.path.join(path_to_checkpoint), map_location=device)
        asr_model.load_state_dict(check_dict["asr_model"])
        tts_model.load_state_dict(check_dict["tts_model"])
        if not fine_tune:
            optim_asr.load_state_dict(check_dict["optimizer"])
            step_counter = check_dict["step_counter"]
            if step_counter > steps:
                print("Desired steps already reached in loaded checkpoint.")
                return
    start_time = time.time()

    while True:
        loss_sum = list()

        tts_model.train()
        asr_model.train()
        for batch in train_loader:
            tokens, mel, tokens_len, mel_len = to_device(batch, device)

            pred = asr_model(mel)
            pred_tts = tts_model(pred.softmax(-1)[:, :, :])

            pred = pred.transpose(0, 1).log_softmax(2)
            loss_ctc = ctc_loss(pred, tokens, mel_len, tokens_len)

            loss = torch.nn.functional.l1_loss(pred_tts, mel)

            factor = 1.
            if step_counter > 2000:
                factor = 0.
            loss = loss + factor * loss_ctc

            optim_asr.zero_grad()
            optim_tts.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(asr_model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(tts_model.parameters(), 1.0)
            optim_asr.step()
            optim_tts.step()

            loss_sum.append(loss.item())

        tts_model.eval()
        asr_model.eval()
        loss_this_epoch = sum(loss_sum) / len(loss_sum)
        if epoch % epochs_per_save == 0:
            torch.save({
                "asr_model": asr_model.state_dict(),
                "tts_model": tts_model.state_dict(),
                "optimizer": optim_asr.state_dict(),
                "step_counter": step_counter,
            },
                os.path.join(save_directory, "checkpoint_{}.pt".format(step_counter)))
            delete_old_checkpoints(save_directory, keep=5)
            if step_counter > steps:
                print("Epoch:        {}".format(epoch))
                print("Total Loss:   {}".format(round(loss_this_epoch, 3)))
                print("Time elapsed: {} Minutes".format(round((time.time() - start_time) / 60)))
                print("Steps:        {}".format(step_counter))
                # DONE
                return
        print("Epoch:        {}".format(epoch))
        print("Total Loss:   {}".format(round(loss_this_epoch, 3)))
        print("Time elapsed: {} Minutes".format(round((time.time() - start_time) / 60)))
        print("Steps:        {}".format(step_counter))
