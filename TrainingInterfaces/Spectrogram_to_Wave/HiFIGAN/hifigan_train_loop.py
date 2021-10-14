import os
import time

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


def train_loop(generator,
               discriminator,
               train_dataset,
               device,
               model_save_dir,
               epochs_per_save=1,
               path_to_checkpoint=None,
               batch_size=16,
               steps=500000):
    torch.backends.cudnn.benchmark = True
    # we have fixed input sizes, so we can enable benchmark mode

    step_counter = 0
    epoch = 0

    mel_l1 = MelSpectrogramLoss().to(device)
    feat_match_criterion = FeatureMatchLoss().to(device)
    discriminator_adv_criterion = DiscriminatorAdversarialLoss().to(device)
    generator_adv_criterion = GeneratorAdversarialLoss().to(device)

    g = generator.to(device)
    d = discriminator.to(device)
    g.train()
    d.train()
    optimizer_g = torch.optim.Adam(g.parameters(), betas=(0.5, 0.9), lr=2.0e-4, weight_decay=0.0)
    scheduler_g = MultiStepLR(optimizer_g, gamma=0.5, milestones=[200000, 400000, 600000, 800000])
    optimizer_d = torch.optim.Adam(d.parameters(), betas=(0.5, 0.9), lr=2.0e-4, weight_decay=0.0)
    scheduler_d = MultiStepLR(optimizer_d, gamma=0.5, milestones=[200000, 400000, 600000, 800000])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=False, drop_last=True, prefetch_factor=4,
                              persistent_workers=False)

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

    while True:

        epoch += 1
        discriminator_losses = list()
        generator_losses = list()

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
            d_outs = d(pred_wave)
            d_gold_outs = d(gold_wave)
            adversarial_loss = generator_adv_criterion(d_outs)
            mel_loss = mel_l1(pred_wave.squeeze(1), gold_wave)
            feature_matching_loss = feat_match_criterion(d_outs, d_gold_outs)
            generator_total_loss = mel_loss * 45 + adversarial_loss * 1 + feature_matching_loss * 2
            optimizer_g.zero_grad()
            generator_total_loss.backward()
            generator_losses.append(float(generator_total_loss))
            torch.nn.utils.clip_grad_norm_(g.parameters(), 10.0)
            optimizer_g.step()
            scheduler_g.step()
            optimizer_g.zero_grad()

            ############################
            #       Discriminator      #
            ############################

            d_outs = d(pred_wave.detach())  # have to recompute unfortunately due to autograd behaviour
            d_gold_outs = d(gold_wave)  # have to recompute unfortunately due to autograd behaviour
            discriminator_loss = discriminator_adv_criterion(d_outs, d_gold_outs)
            optimizer_d.zero_grad()
            discriminator_loss.backward()
            discriminator_losses.append(float(discriminator_loss))
            torch.nn.utils.clip_grad_norm_(d.parameters(), 10.0)
            optimizer_d.step()
            scheduler_d.step()
            optimizer_d.zero_grad()

        ##########################
        #     Epoch Complete     #
        ##########################

        if epoch % epochs_per_save == 0:
            torch.save({
                "generator": g.state_dict(),
                "discriminator": d.state_dict(),
                "generator_optimizer": optimizer_g.state_dict(),
                "discriminator_optimizer": optimizer_d.state_dict(),
                "generator_scheduler": scheduler_g.state_dict(),
                "discriminator_scheduler": scheduler_d.state_dict(),
                "step_counter": step_counter
            }, os.path.join(model_save_dir, "checkpoint_{}.pt".format(step_counter)))
            delete_old_checkpoints(model_save_dir, keep=5)
            if step_counter > steps:
                # DONE
                return

        print("Epoch:              {}".format(epoch + 1))
        print("Time elapsed:       {} Minutes".format(round((time.time() - start_time) / 60), 2))
        print("Steps:              {}".format(step_counter))
        print("Generator Loss:     {}".format(sum(generator_losses) / len(generator_losses)))
        print("Discriminator Loss: {}".format(sum(discriminator_losses) / len(discriminator_losses)))
