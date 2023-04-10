import os
import random
import time

import torch
import torch.multiprocessing
import wandb
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from TrainingInterfaces.Spectrogram_to_Embedding.StyleEmbedding import StyleEmbedding
from TrainingInterfaces.Text_to_Spectrogram.ToucanTTS.SpectrogramDiscriminator import SpectrogramDiscriminator
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
               postnet_start_steps,
               use_discriminator
               ):
    """
    see train loop arbiter for explanations of the arguments
    """
    net = net.to(device)
    if use_discriminator:
        discriminator = SpectrogramDiscriminator().to(device)

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
    if use_discriminator:
        optimizer = torch.optim.Adam(list(net.parameters()) + list(discriminator.parameters()), lr=lr)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = WarmupScheduler(optimizer, peak_lr=lr, warmup_steps=warmup_steps, max_steps=steps)
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
    start_time = time.time()
    while True:
        net.train()
        epoch += 1
        l1_losses_total = list()
        glow_losses_total = list()
        duration_losses_total = list()
        pitch_losses_total = list()
        energy_losses_total = list()
        generator_losses_total = list()
        discriminator_losses_total = list()

        for batch in tqdm(train_loader):
            train_loss = 0.0
            style_embedding = style_embedding_function(batch_of_spectrograms=batch[2].to(device),
                                                       batch_of_spectrogram_lengths=batch[3].to(device))

            l1_loss, duration_loss, pitch_loss, energy_loss, glow_loss, generated_spectrograms = net(
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

            if use_discriminator:
                discriminator_loss, generator_loss = calc_gan_outputs(real_spectrograms=batch[2].to(device),
                                                                      fake_spectrograms=generated_spectrograms,
                                                                      spectrogram_lengths=batch[3].to(device),
                                                                      discriminator=discriminator)
                if not torch.isnan(discriminator_loss):
                    train_loss = train_loss + discriminator_loss
                if not torch.isnan(generator_loss):
                    train_loss = train_loss + generator_loss
                discriminator_losses_total.append(discriminator_loss.item())
                generator_losses_total.append(generator_loss.item())

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
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=False)
            optimizer.step()
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
            "scheduler"   : scheduler.state_dict(),
            "default_emb" : default_embedding,
        }, os.path.join(save_directory, "checkpoint_{}.pt".format(step_counter)))
        delete_old_checkpoints(save_directory, keep=5)

        print("\nEpoch:                  {}".format(epoch))
        print("Time elapsed:           {} Minutes".format(round((time.time() - start_time) / 60)))
        print("Reconstruction Loss:    {}".format(round(sum(l1_losses_total) / len(l1_losses_total), 3)))
        print("Steps:                  {}\n".format(step_counter))
        if use_wandb:
            wandb.log({
                "l1_loss"      : round(sum(l1_losses_total) / len(l1_losses_total), 5),
                "duration_loss": round(sum(duration_losses_total) / len(duration_losses_total), 5),
                "pitch_loss"   : round(sum(pitch_losses_total) / len(pitch_losses_total), 5),
                "energy_loss"  : round(sum(energy_losses_total) / len(energy_losses_total), 5),
                "glow_loss"    : round(sum(glow_losses_total) / len(glow_losses_total), 5) if len(glow_losses_total) != 0 else None,
            }, step=step_counter)
            if use_discriminator:
                wandb.log({
                    "critic_loss"   : round(sum(discriminator_losses_total) / len(discriminator_losses_total), 5),
                    "generator_loss": round(sum(generator_losses_total) / len(generator_losses_total), 5),
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

        if step_counter > 3 * postnet_start_steps:
            # Run manual SWA (torch builtin doesn't work unfortunately due to the use of weight norm in the postflow)
            checkpoint_paths = get_n_recent_checkpoints_paths(checkpoint_dir=save_directory, n=2)
            averaged_model, default_embed = average_checkpoints(checkpoint_paths, load_func=load_net_toucan)
            save_model_for_use(model=averaged_model, default_embed=default_embed, name=os.path.join(save_directory, "best.pt"))
            check_dict = torch.load(os.path.join(save_directory, "best.pt"), map_location=device)
            net.load_state_dict(check_dict["model"])

        if step_counter > steps:
            return  # DONE

        net.train()


def calc_gan_outputs(real_spectrograms, fake_spectrograms, spectrogram_lengths, discriminator):
    # we have signals with lots of padding and different shapes, so we need to extract fixed size windows first.
    fake_window, real_window = get_random_window(fake_spectrograms, real_spectrograms, spectrogram_lengths)
    # now we have windows that are [batch_size, 200, 80]
    critic_loss = discriminator.calc_discriminator_loss(fake_window.unsqueeze(1), real_window.unsqueeze(1))
    generator_loss = discriminator.calc_generator_feedback(fake_window.unsqueeze(1), real_window.unsqueeze(1))
    critic_loss = critic_loss
    generator_loss = generator_loss
    return critic_loss, generator_loss


def get_random_window(generated_sequences, real_sequences, lengths):
    """
    This will return a randomized but consistent window of each that can be passed to the discriminator
    Suboptimal runtime because of a loop, should not be too bad, but a fix would be nice.
    """
    generated_windows = list()
    real_windows = list()
    window_size = 100  # corresponds to 1.6 seconds of audio in real time

    for end_index, generated, real in zip(lengths.squeeze(), generated_sequences, real_sequences):

        length = end_index
        real_spec_unpadded = real[:end_index]
        fake_spec_unpadded = generated[:end_index]
        while length < window_size:
            real_spec_unpadded = real_spec_unpadded.repeat((2, 1))
            fake_spec_unpadded = fake_spec_unpadded.repeat((2, 1))
            length = length * 2

        max_start = length - window_size
        start = random.randint(0, max_start)

        generated_windows.append(fake_spec_unpadded[start:start + window_size].unsqueeze(0))
        real_windows.append(real_spec_unpadded[start:start + window_size].unsqueeze(0))
    return torch.cat(generated_windows, dim=0), torch.cat(real_windows, dim=0)
