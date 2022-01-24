import os
import time

import auraloss
import torch
import torch.multiprocessing
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from TrainingInterfaces.Spectrogram_to_Wave.HiFIGAN.AdversarialLosses import DiscriminatorAdversarialLoss
from TrainingInterfaces.Spectrogram_to_Wave.HiFIGAN.AdversarialLosses import GeneratorAdversarialLoss
from TrainingInterfaces.Spectrogram_to_Wave.HiFIGAN.FeatureMatchingLoss import FeatureMatchLoss
from TrainingInterfaces.Spectrogram_to_Wave.HiFIGAN.MelSpectrogramLoss import MelSpectrogramLoss
from Utility.utils import delete_old_checkpoints
from Utility.utils import get_most_recent_checkpoint


def train_loop(generator,
               discriminator,
               train_dataset,
               device,
               model_save_dir,
               epochs_per_save=1,
               path_to_checkpoint=None,
               batch_size=32,
               steps=2500000,
               resume=False,
               use_signal_processing_losses=False  # https://github.com/csteinmetz1/auraloss remember to cite if used
               ):
    torch.backends.cudnn.benchmark = True
    # we have fixed input sizes, so we can enable benchmark mode

    step_counter = 0
    epoch = 0

    mel_l1 = MelSpectrogramLoss().to(device)
    feat_match_criterion = FeatureMatchLoss().to(device)
    discriminator_adv_criterion = DiscriminatorAdversarialLoss().to(device)
    generator_adv_criterion = GeneratorAdversarialLoss().to(device)

    signal_processing_loss_functions = list()
    if use_signal_processing_losses:
        signal_processing_loss_functions.append(auraloss.time.SNRLoss().to(device))
        signal_processing_loss_functions.append(auraloss.time.SISDRLoss().to(device))

    g = generator.to(device)
    d = discriminator.to(device)
    g.train()
    d.train()
    optimizer_g = torch.optim.Adam(g.parameters(), betas=(0.5, 0.9), lr=2.0e-4, weight_decay=0.0)
    scheduler_g = MultiStepLR(optimizer_g, gamma=0.5, milestones=[200000, 400000, 600000, 800000])
    optimizer_d = torch.optim.Adam(d.parameters(), betas=(0.5, 0.9), lr=2.0e-4, weight_decay=0.0)
    scheduler_d = MultiStepLR(optimizer_d, gamma=0.5, milestones=[200000, 400000, 600000, 800000])

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=16,
                              pin_memory=True,
                              drop_last=True,
                              prefetch_factor=8,
                              persistent_workers=True)

    if resume:
        path_to_checkpoint = get_most_recent_checkpoint(checkpoint_dir=model_save_dir)
    if path_to_checkpoint is not None:
        check_dict = torch.load(path_to_checkpoint, map_location=device)
        optimizer_g.load_state_dict(check_dict["generator_optimizer"])
        optimizer_d.load_state_dict(check_dict["discriminator_optimizer"])
        scheduler_g.load_state_dict(check_dict["generator_scheduler"])
        scheduler_d.load_state_dict(check_dict["discriminator_scheduler"])
        g.load_state_dict(check_dict["generator"])
        d.load_state_dict(check_dict["discriminator"])
        step_counter = check_dict["step_counter"]

    start_time = time.time()

    for _ in range(steps):

        epoch += 1
        discriminator_losses = list()
        generator_losses = list()
        mel_losses = list()
        feat_match_losses = list()
        adversarial_losses = list()
        signal_processing_losses = list()

        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        for datapoint in tqdm(train_loader):
            step_counter += 1

            ############################
            #         Generator        #
            ############################

            gold_wave = datapoint[0].to(device).unsqueeze(1)
            melspec = datapoint[1].to(device)
            pred_wave = g(melspec)
            signal_loss = 0.0
            if use_signal_processing_losses:
                for sl in signal_processing_loss_functions:
                    signal_loss += sl(pred_wave, gold_wave)
                signal_processing_losses.append(signal_loss.item())
            d_outs = d(pred_wave)
            d_gold_outs = d(gold_wave)
            if step_counter > 10000:  # a little bit of warmup helps, but it's not that important
                adversarial_loss = generator_adv_criterion(d_outs)
            else:
                adversarial_loss = 0.0
            mel_loss = mel_l1(pred_wave.squeeze(1), gold_wave)
            feature_matching_loss = feat_match_criterion(d_outs, d_gold_outs)
            generator_total_loss = mel_loss * 40.0 + adversarial_loss * 4.0 + feature_matching_loss * 0.3 + signal_loss
            optimizer_g.zero_grad()
            generator_total_loss.backward()
            generator_losses.append(generator_total_loss.item())
            mel_losses.append(mel_loss.item() * 40.0)
            feat_match_losses.append(feature_matching_loss.item() * 0.3)
            adversarial_losses.append(adversarial_loss.item() * 4.0)
            torch.nn.utils.clip_grad_norm_(g.parameters(), 10.0)
            optimizer_g.step()
            scheduler_g.step()
            optimizer_g.zero_grad()

            ############################
            #       Discriminator      #
            ############################

            # wasserstein seems appropriate, because the discriminator learns much much quicker
            if step_counter > 10000 and step_counter % 2 == 0:
                d_outs = d(pred_wave.detach())  # have to recompute unfortunately due to autograd behaviour
                d_gold_outs = d(gold_wave)  # have to recompute unfortunately due to autograd behaviour
                discriminator_loss = discriminator_adv_criterion(d_outs, d_gold_outs)
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
            torch.save({
                "generator"              : g.state_dict(),
                "discriminator"          : d.state_dict(),
                "generator_optimizer"    : optimizer_g.state_dict(),
                "discriminator_optimizer": optimizer_d.state_dict(),
                "generator_scheduler"    : scheduler_g.state_dict(),
                "discriminator_scheduler": scheduler_d.state_dict(),
                "step_counter"           : step_counter
                }, os.path.join(model_save_dir, "checkpoint_{}.pt".format(step_counter)))
            delete_old_checkpoints(model_save_dir, keep=5)

        print("Epoch:              {}".format(epoch + 1))
        print("Time elapsed:       {} Minutes".format(round((time.time() - start_time) / 60)))
        print("Steps:              {}".format(step_counter))
        print("Generator Loss:     {}".format(round(sum(generator_losses) / len(generator_losses), 3)))
        print("    Mel Loss:       {}".format(round(sum(mel_losses) / len(mel_losses), 3)))
        print("    FeatMatch Loss: {}".format(round(sum(feat_match_losses) / len(feat_match_losses), 3)))
        if use_signal_processing_losses:
            print("    SigProc Loss:   {}".format(round(sum(signal_processing_losses) / len(signal_processing_losses), 3)))
        print("    Adv Loss:       {}".format(round(sum(adversarial_losses) / len(adversarial_losses), 3)))
        print("Discriminator Loss: {}".format(round(sum(discriminator_losses) / len(discriminator_losses), 3)))
