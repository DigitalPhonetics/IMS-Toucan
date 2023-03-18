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
from Utility.WarmupScheduler import ToucanWarmupScheduler as WarmupScheduler
from Utility.utils import delete_old_checkpoints
from Utility.utils import get_most_recent_checkpoint
from Utility.utils import plot_progress_spec_toucantts
from run_weight_averaging import average_checkpoints
from run_weight_averaging import get_n_recent_checkpoints_paths
from run_weight_averaging import load_net_toucan
from run_weight_averaging import save_model_for_use


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
               batch_size,
               lang,
               lr,
               warmup_steps,
               path_to_checkpoint,
               path_to_embed_model,
               fine_tune,
               resume,
               steps,
               use_wandb,
               postnet_start_steps
               ):
    """
    see train loop arbiter for explanations of the arguments
    """
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
                              num_workers=12 if os.cpu_count() > 12 else max(os.cpu_count() - 2, 1),
                              pin_memory=True,
                              shuffle=True,
                              prefetch_factor=2,
                              collate_fn=collate_and_pad,
                              persistent_workers=True)
    step_counter = 0
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = WarmupScheduler(optimizer, peak_lr=lr, warmup_steps=warmup_steps, max_steps=steps)
    grad_scaler = GradScaler()
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
            grad_scaler.load_state_dict(check_dict["scaler"])
    start_time = time.time()
    while True:
        net.train()
        epoch += 1
        l1_losses_total = list()
        glow_losses_total = list()
        duration_losses_total = list()
        pitch_losses_total = list()
        energy_losses_total = list()

        for batch in tqdm(train_loader):
            train_loss = 0.0
            with autocast():
                style_embedding = style_embedding_function(batch_of_spectrograms=batch[2].to(device),
                                                           batch_of_spectrogram_lengths=batch[3].to(device))

                l1_loss, duration_loss, pitch_loss, energy_loss, glow_loss = net(
                    text_tensors=batch[0].to(device),
                    text_lengths=batch[1].to(device),
                    gold_speech=batch[2].to(device),
                    speech_lengths=batch[3].to(device),
                    gold_durations=batch[4].to(device),
                    gold_pitch=batch[6].to(device),  # mind the switched order
                    gold_energy=batch[5].to(device),  # mind the switched order
                    utterance_embedding=style_embedding,
                    lang_ids=batch[8].to(device),
                    return_mels=False,
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
                        glow_losses_total.append(glow_loss.item())

            l1_losses_total.append(l1_loss.item())
            duration_losses_total.append(duration_loss.item())
            pitch_losses_total.append(pitch_loss.item())
            energy_losses_total.append(energy_loss.item())

            optimizer.zero_grad()
            grad_scaler.scale(train_loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=False)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            scheduler.step()
            step_counter += 1

        # EPOCH IS OVER
        net.eval()
        style_embedding_function.eval()
        default_embedding = style_embedding_function(
            batch_of_spectrograms=train_dataset[0][2].unsqueeze(0).to(device),
            batch_of_spectrogram_lengths=train_dataset[0][3].unsqueeze(0).to(device)).squeeze()
        torch.save({
            "model"       : net.state_dict(),
            "optimizer"   : optimizer.state_dict(),
            "step_counter": step_counter,
            "scaler"      : grad_scaler.state_dict(),
            "scheduler"   : scheduler.state_dict(),
            "default_emb" : default_embedding,
        }, os.path.join(save_directory, "checkpoint_{}.pt".format(step_counter)))
        delete_old_checkpoints(save_directory, keep=5)

        print("Epoch:              {}".format(epoch))
        print("Time elapsed:       {} Minutes".format(round((time.time() - start_time) / 60)))
        print("Steps:              {}".format(step_counter))
        if use_wandb:
            wandb.log({
                "l1_loss"      : round(sum(l1_losses_total) / len(l1_losses_total), 5),
                "duration_loss": round(sum(duration_losses_total) / len(duration_losses_total), 5),
                "pitch_loss"   : round(sum(pitch_losses_total) / len(pitch_losses_total), 5),
                "energy_loss"  : round(sum(energy_losses_total) / len(energy_losses_total), 5),
                "glow_loss"    : round(sum(glow_losses_total) / len(glow_losses_total), 5) if len(glow_losses_total) != 0 else None,
                "Steps"        : step_counter,
            })

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
                })
                if step_counter > postnet_start_steps:
                    wandb.log({
                        "progress_plot_after": wandb.Image(path_to_most_recent_plot_after)
                    })
        except IndexError:
            print("generating progress plots failed.")

        if step_counter > steps:
            # DONE
            return

        if step_counter > 2 * postnet_start_steps:
            # Run manual SWA (torch builtin doesn't work unfortunately due to the use of weight norm in the postflow)
            checkpoint_paths = get_n_recent_checkpoints_paths(checkpoint_dir=save_directory, n=3)
            averaged_model, default_embed = average_checkpoints(checkpoint_paths, load_func=load_net_toucan)
            save_model_for_use(model=averaged_model, default_embed=default_embed, name=os.path.join(save_directory, "best.pt"))
            check_dict = torch.load(os.path.join(save_directory, "best.pt"), map_location=device)
            net.load_state_dict(check_dict["model"])

        net.train()
