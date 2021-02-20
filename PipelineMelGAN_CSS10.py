"""
Train an autoregressive Transformer TTS model on the german single speaker dataset by Hokuspokus
"""

import json
import os
import random
import time
import warnings

import torch
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
    start_time = time.time()
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
    batch_counter = 0
    criterion = MultiResolutionSTFTLoss().to(device)
    discriminator_criterion = torch.nn.MSELoss().to(device)
    g = generator.to(device)
    d = discriminator.to(device)
    g.train()
    d.train()
    optimizer_g = torch.optim.Adam(g.parameters())
    optimizer_d = torch.optim.Adam(d.parameters())
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batchsize,
                              shuffle=True,
                              num_workers=4)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=batchsize,
                              shuffle=False,
                              num_workers=4)
    for epoch in range(epochs):
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        for datapoint in train_loader:
            batch_counter += 1
            ############################
            #         Generator        #
            ############################
            gold_wave = datapoint[0].to(device)
            melspec = datapoint[1].to(device)
            pred_wave = g(melspec).squeeze(1)
            print(len(pred_wave[0]))
            print(len(gold_wave[0]))
            spectral_loss, magnitude_loss = criterion(pred_wave, gold_wave)
            train_losses["multi_res_spectral_convergence"].append(float(spectral_loss))
            train_losses["multi_res_log_stft_mag"].append(float(magnitude_loss))
            if batch_counter > generator_warmup_steps:  # generator needs warmup
                adversarial_loss = 0.0
                discriminator_outputs = d(pred_wave)
                for output in discriminator_outputs:
                    adversarial_loss += discriminator_criterion(output, 1.0) / len(discriminator_outputs)
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
            if batch_counter > generator_warmup_steps:  # generator needs warmup
                new_pred = g(melspec).detach()
                discriminator_mse_loss = torch.Tensor(0.0)
                discriminator_outputs = d(new_pred)
                for output in discriminator_outputs:
                    # fake loss
                    discriminator_mse_loss += discriminator_criterion(output, 0.0) / len(discriminator_outputs)
                discriminator_outputs = d(gold_wave)
                for output in discriminator_outputs:
                    # real loss
                    discriminator_mse_loss += discriminator_criterion(output, 1.0) / len(discriminator_outputs)
                train_losses["discriminator_mse"].append(float(discriminator_mse_loss))
                # discriminator step time
                optimizer_d.zero_grad()
                discriminator_mse_loss.backward()
                optimizer_d.step()
                optimizer_d.zero_grad()
            else:
                train_losses["discriminator_mse"].append(0.0)

            torch.cuda.empty_cache()
            print("Step {}".format(batch_counter))

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
                spectral_loss, magnitude_loss = criterion(pred_wave, gold_wave)
                valid_losses["multi_res_spectral_convergence"].append(float(spectral_loss))
                valid_losses["multi_res_log_stft_mag"].append(float(magnitude_loss))
                if batch_counter > generator_warmup_steps:  # generator needs warmup
                    adversarial_loss = 0.0
                    discriminator_outputs = d(pred_wave)
                    for output in discriminator_outputs:
                        adversarial_loss += discriminator_criterion(output, 1.0) / len(discriminator_outputs)
                    valid_losses["adversarial"].append(float(adversarial_loss))
                    generator_total_loss = spectral_loss + magnitude_loss + adversarial_loss
                else:
                    valid_losses["adversarial"].append(0.0)
                    generator_total_loss = spectral_loss + magnitude_loss
                valid_losses["generator_total"].append(float(generator_total_loss))
                if batch_counter > generator_warmup_steps:  # generator needs warmup
                    new_pred = g(melspec).detach()
                    discriminator_mse_loss = 0.0
                    discriminator_outputs = d(new_pred)
                    for output in discriminator_outputs:
                        # fake loss
                        discriminator_mse_loss += discriminator_criterion(output, 0.0) / len(discriminator_outputs)
                    discriminator_outputs = d(gold_wave)
                    for output in discriminator_outputs:
                        # real loss
                        discriminator_mse_loss += discriminator_criterion(output, 1.0) / len(discriminator_outputs)
                    valid_losses["discriminator_mse"].append(float(discriminator_mse_loss))
            valid_gen_mean_epoch_loss = sum(valid_losses["generator_total"][-len(valid_dataset)]) / len(valid_dataset)
            if val_loss_highscore > valid_gen_mean_epoch_loss and batch_counter > generator_warmup_steps:
                # only then it gets interesting
                val_loss_highscore = valid_gen_mean_epoch_loss
                torch.save({"generator": g.state_dict(),
                            "discriminator": d.state_dict(),
                            "generator_optimizer": optimizer_g.state_dict(),
                            "discriminator_optimizer": optimizer_d.state_dict()},
                           os.path.join(model_save_dir,
                                        "checkpoint_{}_{}.pt".format(round(valid_gen_mean_epoch_loss, 4),
                                                                     batch_counter)))
            print("Epoch:                 {}".format(epoch + 1))
            print("Valid GeneratorLoss:   {}".format(valid_gen_mean_epoch_loss))
            print("Time elapsed:          {} Minutes".format(round((time.time() - start_time) / 60), 2))

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


if __name__ == '__main__':
    print("Preparing")
    fl = get_file_list()
    device = torch.device("cpu")
    train_dataset = MelGANDataset(list_of_paths=fl[:-100])
    valid_dataset = MelGANDataset(list_of_paths=fl[-100:])
    generator = MelGANGenerator()
    generator.reset_parameters()
    multi_scale_discriminator = MelGANMultiScaleDiscriminator()
    if not os.path.exists("Models/MelGAN/SingleSpeaker/CSS10"):
        os.makedirs("Models/MelGAN/SingleSpeaker/CSS10")
    print("Training model")
    train_loop(batchsize=16,
               epochs=10,  # for testing
               generator=generator,
               discriminator=multi_scale_discriminator,
               train_dataset=train_dataset,
               valid_dataset=valid_dataset,
               device=device,
               generator_warmup_steps=100)  # for testing
