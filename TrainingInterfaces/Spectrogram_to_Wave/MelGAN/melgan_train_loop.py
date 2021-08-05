import json
import os
import time

import torch
import torch.multiprocessing
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from TrainingInterfaces.Spectrogram_to_Wave.MelGAN.MultiResolutionSTFTLoss import MultiResolutionSTFTLoss
from Utility.RAdam import RAdam
from Utility.utils import delete_old_checkpoints


def train_loop(generator,
               discriminator,
               train_dataset,
               device,
               model_save_dir,
               generator_warmup_steps=100000,
               epochs_per_save=20,
               path_to_checkpoint=None,
               batch_size=16,
               steps=2000000):
    torch.backends.cudnn.benchmark = True
    # we have fixed input sizes, so we can enable benchmark mode

    train_losses = dict()
    train_losses["adversarial"] = list()
    train_losses["multi_res_spectral_convergence"] = list()
    train_losses["multi_res_log_stft_mag"] = list()
    train_losses["generator_total"] = list()
    train_losses["discriminator_mse"] = list()

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
        with open(os.path.join(model_save_dir, "train_loss.json"), "r") as tl:
            train_losses = json.load(tl)
        check_dict = torch.load(path_to_checkpoint, map_location=device)
        optimizer_g.load_state_dict(check_dict["generator_optimizer"])
        optimizer_d.load_state_dict(check_dict["discriminator_optimizer"])
        scheduler_g.load_state_dict(check_dict["generator_scheduler"])
        scheduler_d.load_state_dict(check_dict["discriminator_scheduler"])
        g.load_state_dict(check_dict["generator"])
        d.load_state_dict(check_dict["discriminator"])
        if "step_counter" in check_dict:
            step_counter = check_dict["step_counter"]
        else:
            # legacy
            step_counter = int(path_to_checkpoint.split(".")[0].split("_")[-1])

    start_time = time.time()

    while True:

        epoch += 1

        train_losses_this_epoch = dict()
        train_losses_this_epoch["adversarial"] = list()
        train_losses_this_epoch["multi_res_spectral_convergence"] = list()
        train_losses_this_epoch["multi_res_log_stft_mag"] = list()
        train_losses_this_epoch["generator_total"] = list()
        train_losses_this_epoch["discriminator_mse"] = list()

        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        for datapoint in tqdm(train_loader):
            step_counter += 1
            ############################
            #         Generator        #
            ############################

            gold_wave = datapoint[0].to(device)
            melspec = datapoint[1].to(device)
            del datapoint
            pred_wave = g(melspec)
            spectral_loss, magnitude_loss = criterion(pred_wave.squeeze(1), gold_wave)
            train_losses_this_epoch["multi_res_spectral_convergence"].append(float(spectral_loss))
            train_losses_this_epoch["multi_res_log_stft_mag"].append(float(magnitude_loss))
            if step_counter > generator_warmup_steps:  # generator needs warmup
                d_outs = d(pred_wave)
                adversarial_loss = 0.0
                for i in range(len(d_outs)):
                    adversarial_loss += discriminator_criterion(d_outs[i][-1], d_outs[i][-1].new_ones(d_outs[i][-1].size()))
                adversarial_loss /= (len(d_outs))
                lambda_a = 4
                if step_counter > 200000:
                    lambda_a = 8
                    if step_counter > 300000:
                        lambda_a = 12
                        # the later into training we get, the more valuable the discriminator feedback becomes
                train_losses_this_epoch["adversarial"].append(float(adversarial_loss * lambda_a))

                generator_total_loss = (spectral_loss + magnitude_loss) * 25 + adversarial_loss * lambda_a
            else:
                train_losses_this_epoch["adversarial"].append(0.0)
                generator_total_loss = (spectral_loss + magnitude_loss) * 25
            train_losses_this_epoch["generator_total"].append(float(generator_total_loss))
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

            if step_counter > generator_warmup_steps:  # generator needs warmup
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
                train_losses_this_epoch["discriminator_mse"].append(float(discriminator_mse_loss))
                # discriminator step time
                optimizer_d.zero_grad()
                discriminator_mse_loss.backward()
                torch.nn.utils.clip_grad_norm_(d.parameters(), 1.0)
                optimizer_d.step()
                scheduler_d.step()
                optimizer_d.zero_grad()
            else:
                train_losses_this_epoch["discriminator_mse"].append(0.0)

        ##########################
        #     Epoch Complete     #
        ##########################

        with torch.no_grad():
            g.eval()
            d.eval()
            if step_counter > generator_warmup_steps and epoch % epochs_per_save == 0:
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

            print("Epoch:        {}".format(epoch + 1))
            print("Time elapsed: {} Minutes".format(round((time.time() - start_time) / 60), 2))
            print("Steps:        {}".format(step_counter))
            g.train()
            d.train()

            # average the losses of the epoch for a smooth plotting experience
            train_losses["adversarial"].append(sum(train_losses_this_epoch["adversarial"]) / len(train_losses_this_epoch["adversarial"]))
            train_losses["multi_res_spectral_convergence"].append(
                sum(train_losses_this_epoch["multi_res_spectral_convergence"]) / len(train_losses_this_epoch["multi_res_spectral_convergence"]))
            train_losses["multi_res_log_stft_mag"].append(
                sum(train_losses_this_epoch["multi_res_log_stft_mag"]) / len(train_losses_this_epoch["multi_res_log_stft_mag"]))
            train_losses["generator_total"].append(sum(train_losses_this_epoch["generator_total"]) / len(train_losses_this_epoch["generator_total"]))
            train_losses["discriminator_mse"].append(sum(train_losses_this_epoch["discriminator_mse"]) / len(train_losses_this_epoch["discriminator_mse"]))
            with open(os.path.join(model_save_dir, "train_loss.json"), 'w') as plotting_data_file:
                json.dump(train_losses, plotting_data_file)
