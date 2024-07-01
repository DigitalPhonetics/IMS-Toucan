import os
import time

import torch
import torch.multiprocessing
import wandb
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from Architectures.Vocoder.AdversarialLoss import discriminator_adv_loss
from Architectures.Vocoder.AdversarialLoss import generator_adv_loss
from Architectures.Vocoder.FeatureMatchingLoss import feature_loss
from Architectures.Vocoder.MelSpecLoss import MelSpectrogramLoss
from Utility.utils import delete_old_checkpoints
from Utility.utils import get_most_recent_checkpoint
from run_weight_averaging import average_checkpoints
from run_weight_averaging import get_n_recent_checkpoints_paths
from run_weight_averaging import load_net_bigvgan


def train_loop(generator,
               discriminator,
               train_dataset,
               device,
               model_save_dir,
               epochs_per_save=1,
               path_to_checkpoint=None,
               batch_size=32,
               epochs=100,
               resume=False,
               generator_steps_per_discriminator_step=5,
               generator_warmup=30000,
               use_wandb=False,
               finetune=False
               ):
    step_counter = 0
    epoch = 0

    mel_l1 = MelSpectrogramLoss().to(device)

    g = generator.to(device)
    d = discriminator.to(device)

    g.train()
    d.train()
    optimizer_g = torch.optim.RAdam(g.parameters(), betas=(0.5, 0.9), lr=0.0005, weight_decay=0.0)
    scheduler_g = MultiStepLR(optimizer_g, gamma=0.5, milestones=[400000, 8000000, 000000, 1000000])
    optimizer_d = torch.optim.RAdam(d.parameters(), betas=(0.5, 0.9), lr=0.00025, weight_decay=0.0)
    scheduler_d = MultiStepLR(optimizer_d, gamma=0.5, milestones=[400000, 8000000, 000000, 1000000])

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=True,
                              drop_last=True,
                              prefetch_factor=2,
                              persistent_workers=True)

    if resume:
        path_to_checkpoint = get_most_recent_checkpoint(checkpoint_dir=model_save_dir)
    if path_to_checkpoint is not None:
        check_dict = torch.load(path_to_checkpoint, map_location=device)

        if not finetune:
            optimizer_g.load_state_dict(check_dict["generator_optimizer"])
            optimizer_d.load_state_dict(check_dict["discriminator_optimizer"])
            scheduler_g.load_state_dict(check_dict["generator_scheduler"])
            scheduler_d.load_state_dict(check_dict["discriminator_scheduler"])
            step_counter = check_dict["step_counter"]
        d.load_state_dict(check_dict["discriminator"])
        g.load_state_dict(check_dict["generator"])

    start_time = time.time()

    for _ in range(epochs):

        epoch += 1
        discriminator_losses = list()
        generator_losses = list()
        mel_losses = list()
        feat_match_losses = list()
        adversarial_losses = list()

        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        for datapoint in tqdm(train_loader):

            ############################
            #         Generator        #
            ############################

            gold_wave = datapoint[0].to(device).unsqueeze(1)
            melspec = datapoint[1].to(device)
            pred_wave, intermediate_wave_upsampled_twice, intermediate_wave_upsampled_once = g(melspec)
            if torch.any(torch.isnan(pred_wave)):
                print("A NaN in the wave! Skipping...")
                continue

            mel_loss = mel_l1(pred_wave.squeeze(1), gold_wave)
            generator_total_loss = mel_loss * 85.0

            if step_counter > generator_warmup + 100:  # a bit of warmup helps, but it's not that important
                d_outs, d_fmaps = d(wave=pred_wave,
                                    intermediate_wave_upsampled_twice=intermediate_wave_upsampled_twice,
                                    intermediate_wave_upsampled_once=intermediate_wave_upsampled_once)
                adversarial_loss = generator_adv_loss(d_outs)
                adversarial_losses.append(adversarial_loss.item())
                generator_total_loss = generator_total_loss + adversarial_loss * 2  # based on own experience

                d_gold_outs, d_gold_fmaps = d(gold_wave)
                feature_matching_loss = feature_loss(d_gold_fmaps, d_fmaps)
                feat_match_losses.append(feature_matching_loss.item())
                generator_total_loss = generator_total_loss + feature_matching_loss

            if torch.isnan(generator_total_loss):
                print("Loss turned to NaN, skipping. The GAN possibly collapsed.")
                continue

            step_counter += 1

            optimizer_g.zero_grad()
            generator_total_loss.backward()
            generator_losses.append(generator_total_loss.item())
            mel_losses.append(mel_loss.item())

            torch.nn.utils.clip_grad_norm_(g.parameters(), 10.0)
            optimizer_g.step()
            scheduler_g.step()
            optimizer_g.zero_grad()

            ############################
            #       Discriminator      #
            ############################

            if step_counter > generator_warmup and step_counter % generator_steps_per_discriminator_step == 0:
                d_outs, d_fmaps = d(wave=pred_wave.detach(),
                                    intermediate_wave_upsampled_twice=intermediate_wave_upsampled_twice.detach(),
                                    intermediate_wave_upsampled_once=intermediate_wave_upsampled_once.detach(),
                                    discriminator_train_flag=True)
                d_gold_outs, d_gold_fmaps = d(gold_wave,
                                              discriminator_train_flag=True)  # have to recompute unfortunately due to autograd behaviour
                discriminator_loss = discriminator_adv_loss(d_gold_outs, d_outs)
                optimizer_d.zero_grad()
                discriminator_loss.backward()
                discriminator_losses.append(discriminator_loss.item())
                torch.nn.utils.clip_grad_norm_(d.parameters(), 10.0)
                optimizer_d.step()
                scheduler_d.step()
                optimizer_d.zero_grad()

        ##########################
        #     Epoch Complete     #
        ##########################

        if epoch % epochs_per_save == 0:
            g.eval()
            torch.save({
                "generator"              : g.state_dict(),
                "discriminator"          : d.state_dict(),
                "generator_optimizer"    : optimizer_g.state_dict(),
                "discriminator_optimizer": optimizer_d.state_dict(),
                "generator_scheduler"    : scheduler_g.state_dict(),
                "discriminator_scheduler": scheduler_d.state_dict(),
                "step_counter"           : step_counter
            }, os.path.join(model_save_dir, "checkpoint_{}.pt".format(step_counter)))
            g.train()
            delete_old_checkpoints(model_save_dir, keep=5)

            checkpoint_paths = get_n_recent_checkpoints_paths(checkpoint_dir=model_save_dir, n=2)
            averaged_model, _ = average_checkpoints(checkpoint_paths, load_func=load_net_bigvgan)
            torch.save(averaged_model.state_dict(), os.path.join(model_save_dir, "best.pt"))

        # LOGGING
        log_dict = dict()
        log_dict["Generator Loss"] = round(sum(generator_losses) / len(generator_losses), 3)
        log_dict["Mel Loss"] = round(sum(mel_losses) / len(mel_losses), 3)
        if len(feat_match_losses) > 0:
            log_dict["Feature Matching Loss"] = round(sum(feat_match_losses) / len(feat_match_losses), 3)
        if len(adversarial_losses) > 0:
            log_dict["Adversarial Loss"] = round(sum(adversarial_losses) / len(adversarial_losses), 3)
        if len(discriminator_losses) > 0:
            log_dict["Discriminator Loss"] = round(sum(discriminator_losses) / len(discriminator_losses), 3)

        print("Time elapsed for this run:   {} Minutes".format(round((time.time() - start_time) / 60)))
        for key in log_dict:
            print(f"{key}: {log_dict[key]}")

        if use_wandb:
            wandb.log(log_dict, step=step_counter)
