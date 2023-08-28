import os
import random
import time

import torch
import torch.multiprocessing
import wandb
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from EmbeddingModel.StyleEmbedding import StyleEmbedding
from TTSTrainingInterfaces.ToucanTTS.CodecDiscriminator import SpectrogramDiscriminator
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
    # Assuming you have a list of tensors with shape [9, l, 1024]
    tensor_list = [datapoint[2] for datapoint in batch]

    max_length = max([tensor.size(1) for tensor in tensor_list])

    # Pad tensors in the list
    padded_tensors = []
    for tensor in tensor_list:
        padding = torch.zeros(tensor.size(0), max_length - tensor.size(1), tensor.size(2))
        padded_tensor = torch.cat([tensor, padding], dim=1)
        padded_tensors.append(padded_tensor)

    # Convert the list of padded tensors to a single tensor
    padded_tensor = torch.stack(padded_tensors)

    return (pad_sequence([datapoint[0] for datapoint in batch], batch_first=True),
            torch.stack([datapoint[1] for datapoint in batch]).squeeze(1),
            padded_tensor,
            torch.stack([datapoint[3] for datapoint in batch]).squeeze(1),
            pad_sequence([datapoint[4] for datapoint in batch], batch_first=True),
            pad_sequence([datapoint[5] for datapoint in batch], batch_first=True),
            pad_sequence([datapoint[6] for datapoint in batch], batch_first=True),
            pad_sequence([datapoint[7] for datapoint in batch], batch_first=True),
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
               use_discriminator,
               train_embed
               ):
    """
    see train loop arbiter for explanations of the arguments
    """
    net = net.to(device)
    if use_discriminator:
        discriminator = SpectrogramDiscriminator().to(device)

    style_embedding_function = StyleEmbedding().to(device)
    if path_to_embed_model is not None:
        check_dict = torch.load(path_to_embed_model, map_location=device)
        style_embedding_function.load_state_dict(check_dict["style_emb_func"])
        if not train_embed:
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
        optimizer = torch.optim.AdamW(list(net.parameters()) + list(discriminator.parameters()), lr=lr)
    else:
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr)

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
        classification_losses_total = list()
        refinement_classification_losses_total = list()
        refinement_mlm_losses_total = list()
        duration_losses_total = list()
        pitch_losses_total = list()
        energy_losses_total = list()
        generator_losses_total = list()
        discriminator_losses_total = list()

        for batch in tqdm(train_loader):
            train_loss = 0.0
            style_embedding = style_embedding_function(batch_of_feature_sequences=batch[7].to(device),
                                                       batch_of_feature_sequence_lengths=batch[3].to(device))
            classification_loss, refinemnt_classification_loss, mlm_loss, duration_loss, pitch_loss, energy_loss, generated_features = net(
                text_tensors=batch[0].to(device),
                text_lengths=batch[1].to(device),
                gold_speech=batch[2].to(device),
                speech_lengths=batch[3].to(device),
                gold_durations=batch[4].to(device),
                gold_pitch=batch[6].to(device),  # mind the switched order
                gold_energy=batch[5].to(device),  # mind the switched order
                utterance_embedding=style_embedding,
                lang_ids=batch[8].to(device),
                return_feats=True,
                codebook_curriculum=(step_counter + warmup_steps * 3) // (warmup_steps * 3)  # TODO this requires tuning
            )

            if use_discriminator:
                discriminator_loss, generator_loss = calc_gan_outputs(real_features=batch[2].to(device),
                                                                      fake_features=generated_features,
                                                                      feature_lengths=batch[3].to(device),
                                                                      discriminator=discriminator)
                if not torch.isnan(discriminator_loss):
                    train_loss = train_loss + discriminator_loss
                if not torch.isnan(generator_loss):
                    train_loss = train_loss + generator_loss
                discriminator_losses_total.append(discriminator_loss.item())
                generator_losses_total.append(generator_loss.item())

            if step_counter % (warmup_steps / 4) == 0 and (path_to_embed_model is None or train_embed) and step_counter < warmup_steps * 2 and style_embedding_function.use_gst:
                # the computationally very expensive style token regularization loss to spread out the vectors
                print("calculating the style token regularization loss. This will take a while.")
                emb_reg_loss = style_embedding_function.gst.calculate_ada4_regularization_loss()
                train_loss = train_loss + emb_reg_loss
            if not torch.isnan(classification_loss):
                train_loss = train_loss + classification_loss
            if mlm_loss is not None:
                train_loss = train_loss + mlm_loss
            if refinemnt_classification_loss is not None:
                train_loss = train_loss + refinemnt_classification_loss
            if not torch.isnan(duration_loss):
                train_loss = train_loss + duration_loss
            if not torch.isnan(pitch_loss):
                train_loss = train_loss + pitch_loss
            if not torch.isnan(energy_loss):
                train_loss = train_loss + energy_loss

            classification_losses_total.append(classification_loss.item())
            if mlm_loss is not None:
                refinement_mlm_losses_total.append(mlm_loss.item())
                refinement_classification_losses_total.append(refinemnt_classification_loss.item())
            duration_losses_total.append(duration_loss.item())
            pitch_losses_total.append(pitch_loss.item())
            energy_losses_total.append(energy_loss.item())

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
            batch_of_feature_sequences=train_dataset[0][7].unsqueeze(0).to(device),
            batch_of_feature_sequence_lengths=train_dataset[0][3].unsqueeze(0).to(device)).squeeze()
        torch.save({
            "model"       : net.state_dict(),
            "optimizer"   : optimizer.state_dict(),
            "step_counter": step_counter,
            "scheduler"   : scheduler.state_dict(),
            "default_emb" : default_embedding,
            "config"      : net.config
        }, os.path.join(save_directory, "checkpoint_{}.pt".format(step_counter)))
        if path_to_embed_model is None or train_embed:
            torch.save({
                "style_emb_func": style_embedding_function.state_dict()
            }, os.path.join(save_directory, "embedding_function.pt"))
        delete_old_checkpoints(save_directory, keep=5)

        print(f"\nEpoch:                  {epoch}")
        print(f"Time elapsed:           {round((time.time() - start_time) / 60)} Minutes")
        print(f"Reconstruction Loss:    {round(sum(classification_losses_total) / len(classification_losses_total), 4)}")
        print(f"Steps:                  {step_counter}\n")
        print(f"Currently training {(step_counter + (warmup_steps // 2)) // (warmup_steps // 2)} codebooks.")

        if use_wandb:
            wandb.log({
                "classification_loss": round(sum(classification_losses_total) / len(classification_losses_total), 5),
                "duration_loss"      : round(sum(duration_losses_total) / len(duration_losses_total), 5),
                "pitch_loss"         : round(sum(pitch_losses_total) / len(pitch_losses_total), 5),
                "energy_loss"        : round(sum(energy_losses_total) / len(energy_losses_total), 5),
            }, step=step_counter)
            if use_discriminator:
                wandb.log({
                    "critic_loss"   : round(sum(discriminator_losses_total) / len(discriminator_losses_total), 5),
                    "generator_loss": round(sum(generator_losses_total) / len(generator_losses_total), 5),
                }, step=step_counter)
            if len(refinement_classification_losses_total) != 0:
                wandb.log({
                    "refinement_classification_loss": round(sum(refinement_classification_losses_total) / len(refinement_classification_losses_total), 5),
                    "language_modelling_loss"       : round(sum(refinement_mlm_losses_total) / len(refinement_mlm_losses_total), 5),
                }, step=step_counter)

        path_to_most_recent_plot = plot_progress_spec_toucantts(net,
                                                                device,
                                                                save_dir=save_directory,
                                                                step=step_counter,
                                                                lang=lang,
                                                                default_emb=default_embedding)
        if use_wandb:
            wandb.log({
                "progress_plot": wandb.Image(path_to_most_recent_plot)
            }, step=step_counter)

        if step_counter > steps * 4 / 5:
            # Run manual SWA (torch builtin doesn't work unfortunately due to the use of weight norm)
            checkpoint_paths = get_n_recent_checkpoints_paths(checkpoint_dir=save_directory, n=2)
            averaged_model, default_embed = average_checkpoints(checkpoint_paths, load_func=load_net_toucan)
            save_model_for_use(model=averaged_model, default_embed=default_embed, name=os.path.join(save_directory, "best.pt"))
            check_dict = torch.load(os.path.join(save_directory, "best.pt"), map_location=device)
            net.load_state_dict(check_dict["model"])

        if step_counter > steps:
            return  # DONE

        net.train()


def calc_gan_outputs(real_features, fake_features, feature_lengths, discriminator):
    # we have signals with lots of padding and different shapes, so we need to extract fixed size windows first.
    fake_window, real_window = get_random_window(fake_features, real_features, feature_lengths)
    # now we have windows that are [batch_size, 200, 80]
    critic_loss = discriminator.calc_discriminator_loss(fake_window.unsqueeze(1), real_window.unsqueeze(1))
    generator_loss = discriminator.calc_generator_feedback(fake_window.unsqueeze(1), real_window.unsqueeze(1))

    return critic_loss * 20, generator_loss


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
