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
from Utility.storage_config import MODELS_DIR
from Utility.utils import delete_old_checkpoints
from Utility.utils import get_most_recent_checkpoint
from Utility.utils import plot_progress_spec


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
               warmup_steps=4000,
               path_to_checkpoint=None,
               path_to_embed_model=os.path.join(MODELS_DIR, "Embedding", "embedding_function.pt"),
               fine_tune=False,
               resume=False,
               phase_1_steps=100000,
               phase_2_steps=100000,
               use_wandb=False):
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
        phase_1_steps: how many steps to train before using any of the cycle objectives
        phase_2_steps: how many steps to train using the cycle objectives
        path_to_embed_model: path to the pretrained embedding function
    """

    steps = phase_1_steps + phase_2_steps

    net = net.to(device)
    style_embedding_function = StyleEmbedding().to(device)
    check_dict = torch.load(path_to_embed_model, map_location=device)
    style_embedding_function.load_state_dict(check_dict["style_emb_func"])
    style_embedding_function.eval()
    style_embedding_function.requires_grad_(False)

    torch.multiprocessing.set_sharing_strategy('file_system')
    train_loader = DataLoader(batch_size=batch_size,
                              dataset=train_dataset,
                              drop_last=True,
                              num_workers=8 if os.cpu_count() > 8 else max(os.cpu_count() - 2, 1),
                              pin_memory=True,
                              shuffle=True,
                              prefetch_factor=8,
                              collate_fn=collate_and_pad,
                              persistent_workers=True)
    step_counter = 0
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = WarmupScheduler(optimizer, warmup_steps=warmup_steps)
    scaler = GradScaler()
    epoch = 0
    if resume:
        path_to_checkpoint = get_most_recent_checkpoint(checkpoint_dir=save_directory)
    if path_to_checkpoint is not None:
        check_dict = torch.load(path_to_checkpoint, map_location=device)
        net.load_state_dict(check_dict["model"])
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
        train_losses_this_epoch = list()
        cycle_losses_this_epoch = list()
        for batch in tqdm(train_loader):
            with autocast():
                if step_counter <= phase_1_steps:
                    # ===============================================
                    # =        PHASE 1: no cycle objective          =
                    # ===============================================
                    style_embedding = style_embedding_function(batch_of_spectrograms=batch[2].to(device),
                                                               batch_of_spectrogram_lengths=batch[3].to(device))

                    train_loss = net(text_tensors=batch[0].to(device),
                                     text_lengths=batch[1].to(device),
                                     gold_speech=batch[2].to(device),
                                     speech_lengths=batch[3].to(device),
                                     gold_durations=batch[4].to(device),
                                     gold_pitch=batch[6].to(device),  # mind the switched order
                                     gold_energy=batch[5].to(device),  # mind the switched order
                                     utterance_embedding=style_embedding,
                                     lang_ids=batch[8].to(device),
                                     return_mels=False)
                    train_losses_this_epoch.append(train_loss.item())

                else:
                    # ================================================
                    # = PHASE 2:     cycle objective is added        =
                    # ================================================
                    style_embedding_function.eval()
                    style_embedding_of_gold, out_list_gold = style_embedding_function(
                        batch_of_spectrograms=batch[2].to(device),
                        batch_of_spectrogram_lengths=batch[3].to(device),
                        return_all_outs=True)

                    train_loss, output_spectrograms = net(text_tensors=batch[0].to(device),
                                                          text_lengths=batch[1].to(device),
                                                          gold_speech=batch[2].to(device),
                                                          speech_lengths=batch[3].to(device),
                                                          gold_durations=batch[4].to(device),
                                                          gold_pitch=batch[6].to(device),  # mind the switched order
                                                          gold_energy=batch[5].to(device),  # mind the switched order
                                                          utterance_embedding=style_embedding_of_gold.detach(),
                                                          lang_ids=batch[8].to(device),
                                                          return_mels=True)
                    style_embedding_function.train()
                    style_embedding_of_predicted, out_list_predicted = style_embedding_function(
                        batch_of_spectrograms=output_spectrograms,
                        batch_of_spectrogram_lengths=batch[3].to(device),
                        return_all_outs=True)

                    cycle_dist = 0
                    for out_gold, out_pred in zip(out_list_gold, out_list_predicted):
                        # essentially feature matching, as is often done in vocoder training,
                        # since we're essentially dealing with a discriminator here.
                        cycle_dist = cycle_dist + torch.nn.functional.l1_loss(out_pred, out_gold.detach())

                    train_losses_this_epoch.append(train_loss.item())
                    cycle_losses_this_epoch.append(cycle_dist.item())
                    train_loss = train_loss + cycle_dist

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
                "model"       : net.state_dict(),
                "optimizer"   : optimizer.state_dict(),
                "step_counter": step_counter,
                "scaler"      : scaler.state_dict(),
                "scheduler"   : scheduler.state_dict(),
                "default_emb" : default_embedding,
            }, os.path.join(save_directory, "checkpoint_{}.pt".format(step_counter)))
            delete_old_checkpoints(save_directory, keep=5)
            path_to_most_recent_plot = plot_progress_spec(net,
                                                          device,
                                                          save_dir=save_directory,
                                                          step=step_counter,
                                                          lang=lang,
                                                          default_emb=default_embedding)
            if use_wandb:
                wandb.log({
                    "progress_plot": wandb.Image(path_to_most_recent_plot)
                })
        print("Epoch:              {}".format(epoch))
        print("Spectrogram Loss:   {}".format(sum(train_losses_this_epoch) / len(train_losses_this_epoch)))
        if len(cycle_losses_this_epoch) != 0:
            print("Cycle Loss:         {}".format(sum(cycle_losses_this_epoch) / len(cycle_losses_this_epoch)))
        print("Time elapsed:       {} Minutes".format(round((time.time() - start_time) / 60)))
        print("Steps:              {}".format(step_counter))
        if use_wandb:
            wandb.log({
                "spectrogram_loss": sum(train_losses_this_epoch) / len(train_losses_this_epoch),
                "cycle_loss"      : sum(cycle_losses_this_epoch) / len(cycle_losses_this_epoch) if len(
                    cycle_losses_this_epoch) != 0 else 0.0,
                "Steps"           : step_counter,
            })
        if step_counter > steps and epoch % epochs_per_save == 0:
            # DONE
            return
        net.train()
