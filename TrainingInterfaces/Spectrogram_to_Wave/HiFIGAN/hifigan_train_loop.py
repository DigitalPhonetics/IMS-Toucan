import os
import time

import torch
import torch.multiprocessing
from TrainingInterfaces.Spectrogram_to_Wave.HiFIGAN.MultiResolutionSTFTLoss import MultiResolutionSTFTLoss
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from Utility.RAdam import RAdam
from Utility.utils import delete_old_checkpoints


def train_loop(generator,
               discriminator,
               train_dataset,
               device,
               model_save_dir,
               epochs_per_save=20,
               path_to_checkpoint=None,
               batch_size=16,
               steps=2000000):
    torch.backends.cudnn.benchmark = True
    # we have fixed input sizes, so we can enable benchmark mode

    step_counter = 0
    epoch = 0

    criterion = MultiResolutionSTFTLoss().to(device)
    discriminator_criterion = torch.nn.MSELoss().to(device)

    g = generator.to(device)
    d = discriminator.to(device)
    g.train()
    d.train()
    optimizer_g = RAdam(g.parameters(), lr=0.0001, eps=1.0e-6, weight_decay=0.0)
    scheduler_g = MultiStepLR(optimizer_g, gamma=0.5, milestones=[200000, 400000, 600000, 800000, 1000000, 1200000, 1400000, 1600000, 1800000, 2000000])
    optimizer_d = RAdam(d.parameters(), lr=0.00005, eps=1.0e-6, weight_decay=0.0)
    scheduler_d = MultiStepLR(optimizer_d, gamma=0.5, milestones=[200000, 400000, 600000, 800000, 1000000, 1200000, 1400000, 1600000, 1800000, 2000000])

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

        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        for datapoint in tqdm(train_loader):
            step_counter += 1

            ############################
            #         Generator        #
            ############################

            gold_wave = datapoint[0].to(device)
            melspec = datapoint[1].to(device)
            pred_wave = g(melspec)
            spectral_loss, magnitude_loss = criterion(pred_wave.squeeze(1), gold_wave)
            d_outs = d(pred_wave)
            adversarial_loss = 0.0
            for i in range(len(d_outs)):
                adversarial_loss += discriminator_criterion(d_outs[i][-1], d_outs[i][-1].new_ones(d_outs[i][-1].size()))
            adversarial_loss /= (len(d_outs))
            generator_total_loss = (spectral_loss + magnitude_loss) * 25 + adversarial_loss
            # generator step time
            optimizer_g.zero_grad()
            generator_total_loss.backward()
            torch.nn.utils.clip_grad_norm_(g.parameters(), 10.0)
            optimizer_g.step()
            scheduler_g.step()
            optimizer_g.zero_grad()

            ############################
            #       Discriminator      #
            ############################

            new_pred = g(melspec).detach()
            fake_loss = 0.0
            d_outs = d(new_pred)
            for i in range(len(d_outs)):
                fake_loss += discriminator_criterion(d_outs[i][-1], d_outs[i][-1].new_zeros(d_outs[i][-1].size()))
            fake_loss /= len(d_outs)
            real_loss = 0.0
            d_outs = d(gold_wave.unsqueeze(1))
            for i in range(len(d_outs)):
                real_loss += discriminator_criterion(d_outs[i][-1], d_outs[i][-1].new_ones(d_outs[i][-1].size()))
            real_loss /= len(d_outs)
            discriminator_mse_loss = fake_loss + real_loss
            # discriminator step time
            optimizer_d.zero_grad()
            discriminator_mse_loss.backward()
            torch.nn.utils.clip_grad_norm_(d.parameters(), 1.0)
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
            if step_counter > steps:
                # DONE
                return

        print("Epoch:        {}".format(epoch + 1))
        print("Time elapsed: {} Minutes".format(round((time.time() - start_time) / 60), 2))
        print("Steps:        {}".format(step_counter))
