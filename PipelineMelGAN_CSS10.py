"""
Train non-autoregressive spectrogram inversion model on the german single speaker dataset by Hokuspokus
"""

import json
import os
import random
import time
import warnings

import torch
from adabound import AdaBound
from torch.utils.data.dataloader import DataLoader

from MelGAN.MelGANDataset import MelGANDataset
from MelGAN.MelGANGenerator import MelGANGenerator
from MelGAN.MelGANMultiScaleDiscriminator import MelGANMultiScaleDiscriminator
from MelGAN.MultiResolutionSTFTLoss import MultiResolutionSTFTLoss

warnings.filterwarnings("ignore")

torch.manual_seed(17)
random.seed(17)


def get_file_list():
    file_list = list()
    with open("Corpora/CSS10/transcript.txt", encoding="utf8") as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            file_list.append("Corpora/CSS10/" + line.split("|")[0])
    return file_list


def train_loop(batchsize=16,
               epochs=10,
               generator=None,
               discriminator=None,
               train_dataset=None,
               valid_dataset=None,
               device=None,
               model_save_dir=None,
               generator_warmup_steps=200000):
    torch.backends.cudnn.benchmark = True
    # we have fixed input sizes, so we can enable benchmark mode

    train_losses = dict()
    train_losses["adversarial"] = list()
    train_losses["multi_res_spectral_convergence"] = list()
    train_losses["multi_res_log_stft_mag"] = list()
    train_losses["generator_total"] = list()
    train_losses["discriminator_mse"] = list()

    valid_losses = dict()
    valid_losses["adversarial"] = list()
    valid_losses["multi_res_spectral_convergence"] = list()
    valid_losses["multi_res_log_stft_mag"] = list()
    valid_losses["generator_total"] = list()
    valid_losses["discriminator_mse"] = list()

    val_loss_highscore = 100.0
    step_counter = 0

    criterion = MultiResolutionSTFTLoss().to(device)
    discriminator_criterion = torch.nn.MSELoss().to(device)

    g = generator.to(device)
    d = discriminator.to(device)
    g.train()
    d.train()
    optimizer_g = AdaBound(g.parameters())
    optimizer_d = AdaBound(d.parameters())

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batchsize,
                              shuffle=True,
                              num_workers=16,
                              pin_memory=True,
                              drop_last=True,
                              prefetch_factor=10)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=1,
                              shuffle=False,
                              num_workers=16,
                              pin_memory=True,
                              drop_last=False,
                              prefetch_factor=10)

    start_time = time.time()

    for epoch in range(epochs):
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        for datapoint in train_loader:
            step_counter += 1
            ############################
            #         Generator        #
            ############################

            gold_wave = datapoint[0].to(device)
            melspec = datapoint[1].to(device)
            pred_wave = g(melspec)
            spectral_loss, magnitude_loss = criterion(pred_wave.squeeze(1), gold_wave)
            train_losses["multi_res_spectral_convergence"].append(float(spectral_loss))
            train_losses["multi_res_log_stft_mag"].append(float(magnitude_loss))
            if step_counter > generator_warmup_steps:  # generator needs warmup
                d_outs = d(pred_wave)
                adversarial_loss = 0.0
                for i in range(len(d_outs)):
                    adversarial_loss += discriminator_criterion(d_outs[i][-1],
                                                                d_outs[i][-1].new_ones(d_outs[i][-1].size()))
                adversarial_loss /= (len(d_outs))
                train_losses["adversarial"].append(float(adversarial_loss))
                generator_total_loss = spectral_loss + magnitude_loss + adversarial_loss
            else:
                train_losses["adversarial"].append(0.0)
                generator_total_loss = spectral_loss + magnitude_loss
            train_losses["generator_total"].append(float(generator_total_loss))
            # generator step time
            optimizer_g.zero_grad()
            generator_total_loss.backward()
            optimizer_g.step()
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
                train_losses["discriminator_mse"].append(float(discriminator_mse_loss))
                # discriminator step time
                optimizer_d.zero_grad()
                discriminator_mse_loss.backward()
                optimizer_d.step()
                optimizer_d.zero_grad()
            else:
                train_losses["discriminator_mse"].append(0.0)
            # print("Step {}".format(step_counter))

        ############################
        #         Evaluate         #
        ############################

        with torch.no_grad():
            g.eval()
            d.eval()
            for datapoint in valid_loader:
                gold_wave = datapoint[0].to(device)
                melspec = datapoint[1].to(device)
                pred_wave = g(melspec)
                spectral_loss, magnitude_loss = criterion(pred_wave.squeeze(1), gold_wave)
                valid_losses["multi_res_spectral_convergence"].append(float(spectral_loss))
                valid_losses["multi_res_log_stft_mag"].append(float(magnitude_loss))
                if step_counter > generator_warmup_steps:  # generator needs warmup
                    d_outs = d(pred_wave)
                    adversarial_loss = 0.0
                    for i in range(len(d_outs)):
                        adversarial_loss += discriminator_criterion(d_outs[i][-1],
                                                                    d_outs[i][-1].new_ones(d_outs[i][-1].size()))
                    adversarial_loss /= (len(d_outs))
                    valid_losses["adversarial"].append(float(adversarial_loss))
                    generator_total_loss = spectral_loss + magnitude_loss + adversarial_loss
                else:
                    valid_losses["adversarial"].append(0.0)
                    generator_total_loss = spectral_loss + magnitude_loss
                valid_losses["generator_total"].append(float(generator_total_loss))
                if step_counter > generator_warmup_steps:  # generator needs warmup
                    fake_loss = 0.0
                    d_outs = d(pred_wave)
                    for i in range(len(d_outs)):
                        fake_loss += discriminator_criterion(d_outs[i][-1],
                                                             d_outs[i][-1].new_zeros(d_outs[i][-1].size()))
                    fake_loss /= len(d_outs)
                    real_loss = 0.0
                    d_outs = d(gold_wave.unsqueeze(1))
                    for i in range(len(d_outs)):
                        real_loss += discriminator_criterion(d_outs[i][-1],
                                                             d_outs[i][-1].new_ones(d_outs[i][-1].size()))
                    real_loss /= len(d_outs)
                    discriminator_mse_loss = fake_loss + real_loss
                    valid_losses["discriminator_mse"].append(float(discriminator_mse_loss))
            valid_gen_mean_epoch_loss = sum(valid_losses["generator_total"][-len(valid_dataset):]) / len(valid_dataset)
            if val_loss_highscore > valid_gen_mean_epoch_loss and step_counter > generator_warmup_steps:
                # only then it gets interesting
                val_loss_highscore = valid_gen_mean_epoch_loss
                torch.save({"generator": g.state_dict(),
                            "discriminator": d.state_dict(),
                            "generator_optimizer": optimizer_g.state_dict(),
                            "discriminator_optimizer": optimizer_d.state_dict()},
                           os.path.join(model_save_dir,
                                        "checkpoint_{}_{}.pt".format(round(valid_gen_mean_epoch_loss, 4),
                                                                     step_counter)))
            print("Epoch:                  {}".format(epoch + 986 + 1))
            print("Valid Generator Loss:   {}".format(valid_gen_mean_epoch_loss))
            print("Time elapsed:           {} Minutes".format(round((time.time() - start_time) / 60), 2))
            print("Steps:                  {}".format(step_counter))

            with open(os.path.join(model_save_dir, "train_loss.json"), 'w') as plotting_data_file:
                json.dump(train_losses, plotting_data_file)
            with open(os.path.join(model_save_dir, "valid_loss.json"), 'w') as plotting_data_file:
                json.dump(valid_losses, plotting_data_file)
            g.train()
            d.train()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def show_model(model):
    print(model)
    print("\n\nNumber of Parameters: {}".format(count_parameters(model)))


def continue_training(batchsize=16,
                      epochs=10,
                      generator=None,
                      discriminator=None,
                      train_dataset=None,
                      valid_dataset=None,
                      device=None,
                      model_save_dir=None,
                      generator_warmup_steps=200000):
    torch.backends.cudnn.benchmark = True
    # we have fixed input sizes, so we can enable benchmark mode

    with open(os.path.join(model_save_dir, "train_loss.json"), 'r') as plotting_data_file:
        train_losses = json.load(plotting_data_file)
    with open(os.path.join(model_save_dir, "valid_loss.json"), 'r') as plotting_data_file:
        valid_losses = json.load(plotting_data_file)

    val_loss_highscore = 100.0
    step_counter = 200868

    criterion = MultiResolutionSTFTLoss().to(device)
    discriminator_criterion = torch.nn.MSELoss().to(device)

    g = generator.to(device)
    d = discriminator.to(device)
    g.load_state_dict(torch.load("Models/MelGAN/SingleSpeaker/CSS10/checkpoint_1.483_200868")["generator"])
    d.load_state_dict(torch.load("Models/MelGAN/SingleSpeaker/CSS10/checkpoint_1.483_200868")["discriminator"])
    g.train()
    d.train()
    optimizer_g = torch.optim.Adam(g.parameters())
    optimizer_d = torch.optim.Adam(d.parameters())
    optimizer_g.load_state_dict(
        torch.load("Models/MelGAN/SingleSpeaker/CSS10/checkpoint_1.483_200868")["generator_optimizer"])
    optimizer_d.load_state_dict(
        torch.load("Models/MelGAN/SingleSpeaker/CSS10/checkpoint_1.483_200868")["discriminator_optimizer"])

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batchsize,
                              shuffle=True,
                              num_workers=16,
                              pin_memory=True,
                              drop_last=True,
                              prefetch_factor=10)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=1,
                              shuffle=False,
                              num_workers=16,
                              pin_memory=True,
                              drop_last=False,
                              prefetch_factor=10)

    start_time = time.time()

    for epoch in range(epochs):
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        for datapoint in train_loader:
            step_counter += 1
            ############################
            #         Generator        #
            ############################

            gold_wave = datapoint[0].to(device)
            melspec = datapoint[1].to(device)
            pred_wave = g(melspec)
            spectral_loss, magnitude_loss = criterion(pred_wave.squeeze(1), gold_wave)
            train_losses["multi_res_spectral_convergence"].append(float(spectral_loss))
            train_losses["multi_res_log_stft_mag"].append(float(magnitude_loss))
            if step_counter > generator_warmup_steps:  # generator needs warmup
                d_outs = d(pred_wave)
                adversarial_loss = 0.0
                for i in range(len(d_outs)):
                    adversarial_loss += discriminator_criterion(d_outs[i][-1],
                                                                d_outs[i][-1].new_ones(d_outs[i][-1].size()))
                adversarial_loss /= (len(d_outs))
                train_losses["adversarial"].append(float(adversarial_loss))
                generator_total_loss = spectral_loss + magnitude_loss + adversarial_loss
            else:
                train_losses["adversarial"].append(0.0)
                generator_total_loss = spectral_loss + magnitude_loss
            train_losses["generator_total"].append(float(generator_total_loss))
            # generator step time
            optimizer_g.zero_grad()
            generator_total_loss.backward()
            optimizer_g.step()
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
                train_losses["discriminator_mse"].append(float(discriminator_mse_loss))
                # discriminator step time
                optimizer_d.zero_grad()
                discriminator_mse_loss.backward()
                optimizer_d.step()
                optimizer_d.zero_grad()
            else:
                train_losses["discriminator_mse"].append(0.0)
            # print("Step {}".format(step_counter))

        ############################
        #         Evaluate         #
        ############################

        with torch.no_grad():
            g.eval()
            d.eval()
            for datapoint in valid_loader:
                gold_wave = datapoint[0].to(device)
                melspec = datapoint[1].to(device)
                pred_wave = g(melspec)
                spectral_loss, magnitude_loss = criterion(pred_wave.squeeze(1), gold_wave)
                valid_losses["multi_res_spectral_convergence"].append(float(spectral_loss))
                valid_losses["multi_res_log_stft_mag"].append(float(magnitude_loss))
                if step_counter > generator_warmup_steps:  # generator needs warmup
                    d_outs = d(pred_wave)
                    adversarial_loss = 0.0
                    for i in range(len(d_outs)):
                        adversarial_loss += discriminator_criterion(d_outs[i][-1],
                                                                    d_outs[i][-1].new_ones(d_outs[i][-1].size()))
                    adversarial_loss /= (len(d_outs))
                    valid_losses["adversarial"].append(float(adversarial_loss))
                    generator_total_loss = spectral_loss + magnitude_loss + adversarial_loss
                else:
                    valid_losses["adversarial"].append(0.0)
                    generator_total_loss = spectral_loss + magnitude_loss
                valid_losses["generator_total"].append(float(generator_total_loss))
                if step_counter > generator_warmup_steps:  # generator needs warmup
                    fake_loss = 0.0
                    d_outs = d(pred_wave)
                    for i in range(len(d_outs)):
                        fake_loss += discriminator_criterion(d_outs[i][-1],
                                                             d_outs[i][-1].new_zeros(d_outs[i][-1].size()))
                    fake_loss /= len(d_outs)
                    real_loss = 0.0
                    d_outs = d(gold_wave.unsqueeze(1))
                    for i in range(len(d_outs)):
                        real_loss += discriminator_criterion(d_outs[i][-1],
                                                             d_outs[i][-1].new_ones(d_outs[i][-1].size()))
                    real_loss /= len(d_outs)
                    discriminator_mse_loss = fake_loss + real_loss
                    valid_losses["discriminator_mse"].append(float(discriminator_mse_loss))
            valid_gen_mean_epoch_loss = sum(valid_losses["generator_total"][-len(valid_dataset):]) / len(valid_dataset)
            if val_loss_highscore > valid_gen_mean_epoch_loss and step_counter > generator_warmup_steps:
                # only then it gets interesting
                val_loss_highscore = valid_gen_mean_epoch_loss
                torch.save({"generator": g.state_dict(),
                            "discriminator": d.state_dict(),
                            "generator_optimizer": optimizer_g.state_dict(),
                            "discriminator_optimizer": optimizer_d.state_dict()},
                           os.path.join(model_save_dir,
                                        "checkpoint_{}_{}.pt".format(round(valid_gen_mean_epoch_loss, 4),
                                                                     step_counter)))
            print("Epoch:                  {}".format(epoch + 986 + 1))
            print("Valid Generator Loss:   {}".format(valid_gen_mean_epoch_loss))
            print("Time elapsed:           {} Minutes".format(round((time.time() - start_time) / 60), 2))
            print("Steps:                  {}".format(step_counter))

            with open(os.path.join(model_save_dir, "train_loss_.json"), 'w') as plotting_data_file:
                json.dump(train_losses, plotting_data_file)
            with open(os.path.join(model_save_dir, "valid_loss_.json"), 'w') as plotting_data_file:
                json.dump(valid_losses, plotting_data_file)
            g.train()
            d.train()


if __name__ == '__main__':
    print("Preparing")
    fl = get_file_list()
    device = torch.device("cuda:2")
    train_dataset = MelGANDataset(list_of_paths=fl[:-100])
    valid_dataset = MelGANDataset(list_of_paths=fl[-100:])
    generator = MelGANGenerator()
    generator.reset_parameters()
    multi_scale_discriminator = MelGANMultiScaleDiscriminator()
    if not os.path.exists("Models/MelGAN/SingleSpeaker/CSS10"):
        os.makedirs("Models/MelGAN/SingleSpeaker/CSS10")
    print("Training model")
    continue_training(batchsize=32,
                      epochs=60000,  # just kill the process at some point
                      generator=generator,
                      discriminator=multi_scale_discriminator,
                      train_dataset=train_dataset,
                      valid_dataset=valid_dataset,
                      device=device,
                      generator_warmup_steps=200000,
                      model_save_dir="Models/MelGAN/SingleSpeaker/CSS10")
