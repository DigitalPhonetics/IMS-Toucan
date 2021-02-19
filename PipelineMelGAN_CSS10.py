"""
Train an autoregressive Transformer TTS model on the german single speaker dataset by Hokuspokus
"""

import json
import os
import random
import time

import torch
import torchviz
from torch.utils.data.dataloader import DataLoader

from MelGAN.MelGANDataset import MelGANDataset
from MelGAN.MelGANGenerator import MelGANGenerator
from MelGAN.MelGANMultiScaleDiscriminator import MelGANMultiScaleDiscriminator
from MelGAN.MultiResolutionSTFTLoss import MultiResolutionSTFTLoss
from TransformerTTS.TransformerTTS import Transformer

torch.manual_seed(17)
random.seed(17)


def get_file_list():
    file_list = list()
    with open("Corpora/CSS10/transcript.txt", encoding="utf8") as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            file_list.append(line.split("|")[0])
    return file_list


def train_loop(batchsize=16,
               epochs=10,
               generator=None,
               discriminator=None,
               train_dataset=None,
               valid_dataset=None,
               device=None):
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
    discriminator_criterion = torch.nn.MSELoss()
    g = generator.to(device)
    d = discriminator.to(device)
    g.train()
    d.train()
    optimizer_g = torch.optim.Adam(g.parameters())
    optimizer_d = torch.optim.Adam(d.parameters())
    train_loader = DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=True, num_workers=4)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batchsize, shuffle=False, num_workers=4)
    for epoch in range(epochs):
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        for datapoint in train_loader:
            batch_counter += 1
            # TODO check if this all works on a per batch basis or if it needs to be split up. Also check if collate needs to pad
            ############################
            #         Generator        #
            ############################
            gold_wave = datapoint[0]
            melspec = datapoint[1]
            pred_wave = g(melspec)
            spectral_loss, magnitude_loss = criterion(pred_wave, gold_wave)
            train_losses["multi_res_spectral_convergence"].append(float(spectral_loss))
            train_losses["multi_res_log_stft_mag"].append(float(magnitude_loss))
            if batch_counter > 200000:  # generator needs warmup
                adversarial_loss = 0.0
                discriminator_outputs = d(pred_wave)
                for output in discriminator_outputs:
                    adversarial_loss += discriminator_criterion(output, 1.0) / len(discriminator_outputs)
                train_losses["adversarial"].append(adversarial_loss)
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
            if batch_counter > 200000:  # generator needs warmup
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
                train_losses["discriminator_mse"].append(float(discriminator_mse_loss))
                # discriminator step time
                optimizer_d.zero_grad()
                discriminator_mse_loss.backward()
                optimizer_d.step()
                optimizer_d.zero_grad()
            else:
                train_losses["discriminator_mse"].append(0.0)

            torch.cuda.empty_cache()

        ############################
        #         Evaluate         #
        ############################

        with torch.no_grad():
            g.eval()
            d.eval()
            for datapoint in valid_loader:
                val_losses = list()
                for validation_datapoint_index in range(len(eval_dataset)):
                    eval_datapoint = eval_dataset[validation_datapoint_index]
                    val_losses.append(net(eval_datapoint[0].unsqueeze(0).to(device),
                                          eval_datapoint[1].to(device),
                                          eval_datapoint[2].unsqueeze(0).to(device),
                                          eval_datapoint[3].to(device)
                                          )[0])
            val_loss = float(sum(val_losses) / len(val_losses))
            if val_loss_highscore > val_loss:
                val_loss_highscore = val_loss
                torch.save({"model": net.state_dict(),
                            "optimizer": optimizer.state_dict()},
                           os.path.join(save_directory,
                                        "checkpoint_{}_{}.pt".format(round(val_loss, 4), sample_counter)))
            print("Epoch:        {}".format(epoch + 1))
            print("Train Loss:   {}".format(sum(train_losses_this_epoch) / len(train_losses_this_epoch)))
            print("Valid Loss:   {}".format(val_loss))
            print("Time elapsed: {} Minutes".format(round((time.time() - start_time) / 60), 2))
            loss_plot[0].append(sum(train_losses_this_epoch) / len(train_losses_this_epoch))
            loss_plot[1].append(val_loss)
            with open(os.path.join(save_directory, "train_val_loss.json"), 'w') as plotting_data_file:
                json.dump(loss_plot, plotting_data_file)
            g.train()
            d.train()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def show_model(model):
    print(model)
    print("\n\nNumber of Parameters: {}".format(count_parameters(model)))


def plot_model():
    trans = Transformer(idim=132, odim=80, spk_embed_dim=128)
    out = trans(text=torch.randint(high=120, size=(1, 23)),
                text_lengths=torch.tensor([23]),
                speech=torch.rand((1, 1234, 80)),
                speech_lengths=torch.tensor([1234]),
                spembs=torch.rand(128).unsqueeze(0))
    torchviz.make_dot(out[0].mean(), dict(trans.named_parameters())).render("transformertts_graph", format="png")


if __name__ == '__main__':
    print("Preparing")
    fl = get_file_list()
    device = torch.device("cpu")
    train_dataset = MelGANDataset(list_of_paths=fl, device=device, type='train')
    valid_dataset = MelGANDataset(list_of_paths=fl, device=device, type='valid')
    generator = MelGANGenerator()
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
               device=device)
