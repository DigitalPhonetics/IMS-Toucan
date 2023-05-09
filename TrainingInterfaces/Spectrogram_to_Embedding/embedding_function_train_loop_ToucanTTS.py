"""
Train this to get the fundamental embedding function.
"""

import os
import time

import torch
import torch.multiprocessing
import torch.multiprocessing
import wandb
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from TrainingInterfaces.Spectrogram_to_Embedding.StyleEmbedding import StyleEmbedding
from Utility.WarmupScheduler import WarmupScheduler
from Utility.utils import delete_old_checkpoints
from Utility.utils import get_most_recent_checkpoint
from Utility.utils import plot_progress_spec_toucantts


def collate_and_pad(batch):
    # text, text_len, speech, speech_len, durations, energy, pitch, utterance condition, language_id
    return (pad_sequence([datapoint[0] for datapoint in batch], batch_first=True),
            torch.stack([datapoint[1] for datapoint in batch]).squeeze(1),
            pad_sequence([datapoint[2] for datapoint in batch], batch_first=True),
            torch.stack([datapoint[3] for datapoint in batch]).squeeze(1),
            pad_sequence([datapoint[4] for datapoint in batch], batch_first=True),
            pad_sequence([datapoint[5] for datapoint in batch], batch_first=True),
            pad_sequence([datapoint[6] for datapoint in batch], batch_first=True),
            None,
            torch.stack([datapoint[8] for datapoint in batch]))


def train_loop(net,
               train_dataset,
               device,
               save_directory,
               batch_size=32,
               epochs_per_save=1,
               lang="en",
               lr=0.0001,
               warmup_steps=8000,
               path_to_checkpoint=None,
               fine_tune=False,
               resume=False,
               steps=500000,
               use_wandb=False,
               postnet_start_steps=9000):
    """
    Args:
        resume: whether to resume from the most recent checkpoint
        warmup_steps: how long the learning rate should increase before it reaches the specified value
        lr: The initial learning rate for the optimiser
        path_to_checkpoint: reloads a checkpoint to continue training from there
        fine_tune: whether to load everything from a checkpoint, or only the model parameters
        lang: language of the synthesis
        net: Model to train
        train_dataset: Pytorch Dataset Object for train data
        device: Device to put the loaded tensors on
        save_directory: Where to save the checkpoints
        batch_size: How many elements should be loaded at once
        epochs_per_save: how many epochs to train in between checkpoints
        steps: how many steps to train before using any of the cycle objectives
    """
    net = net.to(device)
    style_embedding_function = StyleEmbedding().to(device)

    torch.multiprocessing.set_sharing_strategy('file_system')
    train_loader = DataLoader(batch_size=batch_size,
                              dataset=train_dataset,
                              drop_last=True,
                              num_workers=8,
                              pin_memory=True,
                              shuffle=True,
                              prefetch_factor=8,
                              collate_fn=collate_and_pad,
                              persistent_workers=True)
    step_counter = 0
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    optimizer.add_param_group({"params": style_embedding_function.parameters()})
    scheduler = WarmupScheduler(optimizer, warmup_steps=warmup_steps)
    scaler = GradScaler()
    epoch = 0
    if resume:
        path_to_checkpoint = get_most_recent_checkpoint(checkpoint_dir=save_directory)
    if path_to_checkpoint is not None:
        check_dict = torch.load(path_to_checkpoint, map_location=device)
        net.load_state_dict(check_dict["model"])
        style_embedding_function.load_state_dict(check_dict["style_emb_func"])
        if not fine_tune:
            optimizer.load_state_dict(check_dict["optimizer"])
            scheduler.load_state_dict(check_dict["scheduler"])
            step_counter = check_dict["step_counter"]
            scaler.load_state_dict(check_dict["scaler"])
    start_time = time.time()
    while True:
        net.train()
        epoch += 1
        optimizer.zero_grad()
        l1_losses_total = list()
        glow_losses_total = list()
        duration_losses_total = list()
        pitch_losses_total = list()
        energy_losses_total = list()
        reg_losses_this_epoch = list()

        if step_counter < 80000:
            # first, the computationally very expensive style token regularization loss to spread out the vectors
            print("calculating the style token regularization loss. This will take a while.")
            reg_loss = style_embedding_function.gst.calculate_ada4_regularization_loss()
            reg_losses_this_epoch.append(reg_loss.item())
            optimizer.zero_grad()
            scaler.scale(reg_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=False)
            scaler.step(optimizer)
            scaler.update()
            del reg_loss

        # then the rest
        optimizer.zero_grad()
        for batch in tqdm(train_loader):
            train_loss = 0.0
            with autocast():
                style_embedding = style_embedding_function(batch_of_spectrograms=batch[2].to(device),
                                                           batch_of_spectrogram_lengths=batch[3].to(device))
                l1_loss, duration_loss, pitch_loss, energy_loss, glow_loss, sent_style_loss, generated_spectrograms = net(
                text_tensors=batch[0].to(device),
                text_lengths=batch[1].to(device),
                gold_speech=batch[2].to(device),
                speech_lengths=batch[3].to(device),
                gold_durations=batch[4].to(device),
                gold_pitch=batch[6].to(device),  # mind the switched order
                gold_energy=batch[5].to(device),  # mind the switched order
                utterance_embedding=style_embedding,
                lang_ids=batch[8].to(device),
                return_mels=True,
                run_glow=step_counter > postnet_start_steps or fine_tune)

                if not torch.isnan(l1_loss):
                    train_loss = train_loss + l1_loss
                if not torch.isnan(duration_loss):
                    train_loss = train_loss + duration_loss
                if not torch.isnan(pitch_loss):
                    train_loss = train_loss + pitch_loss
                if not torch.isnan(energy_loss):
                    train_loss = train_loss + energy_loss
                if glow_loss is not None:
                    if step_counter > postnet_start_steps and not torch.isnan(glow_loss):
                        train_loss = train_loss + glow_loss

                l1_losses_total.append(l1_loss.item())
                duration_losses_total.append(duration_loss.item())
                pitch_losses_total.append(pitch_loss.item())
                energy_losses_total.append(energy_loss.item())
                if step_counter > postnet_start_steps + 500 or fine_tune:
                    # start logging late so the magnitude difference is smaller
                    glow_losses_total.append(glow_loss.item())

            optimizer.zero_grad()
            scaler.scale(train_loss).backward()
            del train_loss
            step_counter += 1
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=False)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        net.eval()
        style_embedding_function.eval()
        if epoch % epochs_per_save == 0:
            default_embedding = style_embedding_function(
                batch_of_spectrograms=train_dataset[0][2].unsqueeze(0).to(device),
                batch_of_spectrogram_lengths=train_dataset[0][3].unsqueeze(0).to(device)).squeeze()
            torch.save({
                "model"         : net.state_dict(),
                "optimizer"     : optimizer.state_dict(),
                "step_counter"  : step_counter,
                "scaler"        : scaler.state_dict(),
                "scheduler"     : scheduler.state_dict(),
                "default_emb"   : default_embedding,
                "style_emb_func": style_embedding_function.state_dict()
            }, os.path.join(save_directory, "checkpoint_{}.pt".format(step_counter)))
            torch.save({
                "style_emb_func": style_embedding_function.state_dict()
            }, os.path.join(save_directory, "embedding_function.pt"))
            delete_old_checkpoints(save_directory, keep=5)

        print("Epoch:              {}".format(epoch))
        print("l1 Loss:   {}".format(round(sum(l1_losses_total) / len(l1_losses_total), 5)))
        if len(reg_losses_this_epoch) != 0:
            print("reg Loss:           {}".format(sum(reg_losses_this_epoch) / len(reg_losses_this_epoch)))
        print("Time elapsed:       {} Minutes".format(round((time.time() - start_time) / 60)))
        print("Steps:              {}".format(step_counter))
        if use_wandb:
            wandb.log({
                "l1_loss"      : round(sum(l1_losses_total) / len(l1_losses_total), 5),
                "duration_loss": round(sum(duration_losses_total) / len(duration_losses_total), 5),
                "pitch_loss"   : round(sum(pitch_losses_total) / len(pitch_losses_total), 5),
                "energy_loss"  : round(sum(energy_losses_total) / len(energy_losses_total), 5),
                "glow_loss"    : round(sum(glow_losses_total) / len(glow_losses_total), 5) if len(glow_losses_total) != 0 else None,
                "basis_reg_loss"  : sum(reg_losses_this_epoch) / len(reg_losses_this_epoch) if len(reg_losses_this_epoch) != 0 else 0.0,
            }, step=step_counter)

        try:
            path_to_most_recent_plot_before, \
            path_to_most_recent_plot_after = plot_progress_spec_toucantts(net,
                                                                          device,
                                                                          save_dir=save_directory,
                                                                          step=step_counter,
                                                                          lang=lang,
                                                                          default_emb=default_embedding,
                                                                          run_postflow=step_counter - 5 > postnet_start_steps)
            if use_wandb:
                wandb.log({
                    "progress_plot_before": wandb.Image(path_to_most_recent_plot_before)
                }, step=step_counter)
                if step_counter > postnet_start_steps or fine_tune:
                    wandb.log({
                        "progress_plot_after": wandb.Image(path_to_most_recent_plot_after)
                    }, step=step_counter)
        except IndexError:
            print("generating progress plots failed.")

        if step_counter > steps and epoch % epochs_per_save == 0:
            # DONE
            return
        net.train()
        style_embedding_function.train()
