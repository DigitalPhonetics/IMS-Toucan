"""
Train an autoregressive Transformer TTS model on the german single speaker dataset by Hokuspokus
"""

import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from MelGAN.MelGANDataset import MelGANDataset
from MelGAN.MelGANGenerator import MelGANGenerator
from MelGAN.MelGANMultiScaleDiscriminator import MelGANMultiScaleDiscriminator
from MelGAN.MultiResolutionSTFTLoss import MultiResolutionSTFTLoss

torch.manual_seed(17)
random.seed(17)


class Collater(object):
    """
    Customized collater for Pytorch DataLoader in training.
    """

    def __init__(self,
                 batch_max_steps=20480,
                 hop_size=256,
                 aux_context_window=2,
                 use_noise_input=False):
        """
        Args:
            batch_max_steps (int): The maximum length of input signal in batch.
            hop_size (int): Hop size of auxiliary features.
            aux_context_window (int): Context window size for auxiliary feature conv.
            use_noise_input (bool): Whether to use noise input.
        """
        if batch_max_steps % hop_size != 0:
            batch_max_steps += -(batch_max_steps % hop_size)
        assert batch_max_steps % hop_size == 0
        self.batch_max_steps = batch_max_steps
        self.batch_max_frames = batch_max_steps // hop_size
        self.hop_size = hop_size
        self.aux_context_window = aux_context_window
        self.use_noise_input = use_noise_input

        # set useful values in random cutting
        self.start_offset = aux_context_window
        self.end_offset = -(self.batch_max_frames + aux_context_window)
        self.mel_threshold = self.batch_max_frames + 2 * aux_context_window

    def __call__(self, batch):
        """
        Convert into batch tensors.
        Args:
            batch (list): list of tuple of the pair of audio and features.
        Returns:
            Tensor: Gaussian noise batch (B, 1, T).
            Tensor: Auxiliary feature batch (B, C, T'), where
                T = (T' - 2 * aux_context_window) * hop_size.
            Tensor: Target signal batch (B, 1, T).
        """
        # check length
        batch = [self._adjust_length(*b) for b in batch if len(b[1]) > self.mel_threshold]
        xs, cs = [b[0] for b in batch], [b[1] for b in batch]
        print(xs)
        print(cs)
        # make batch with random cut
        c_lengths = [len(c) for c in cs]
        start_frames = np.array([np.random.randint(
            self.start_offset, cl + self.end_offset) for cl in c_lengths])
        x_starts = start_frames * self.hop_size
        x_ends = x_starts + self.batch_max_steps
        c_starts = start_frames - self.aux_context_window
        c_ends = start_frames + self.batch_max_frames + self.aux_context_window
        y_batch = [x[start: end] for x, start, end in zip(xs, x_starts, x_ends)]
        c_batch = [c[start: end] for c, start, end in zip(cs, c_starts, c_ends)]

        print(torch.tensor(c_batch, dtype=torch.float))

        # convert each batch to tensor, assume that each item in batch has the same length
        y_batch = torch.tensor(y_batch, dtype=torch.float).unsqueeze(1)  # (B, 1, T)
        c_batch = torch.tensor(c_batch, dtype=torch.float).transpose(2, 1)  # (B, C, T')

        # make input noise signal batch tensor
        if self.use_noise_input:
            z_batch = torch.randn(y_batch.size())  # (B, 1, T)
            return (z_batch, c_batch), y_batch
        else:
            return (c_batch,), y_batch

    def _adjust_length(self, x, c):
        """
        Adjust the audio and feature lengths.
        Note:
            Basically we assume that the length of x and c are adjusted
            through preprocessing stage, but if we use other library processed
            features, this process will be needed.
        """
        if len(x) < len(c) * self.hop_size:
            x = np.pad(x, (0, len(c) * self.hop_size - len(x)), mode="edge")
        # check the length is valid
        assert len(x) == len(c) * self.hop_size
        return x, c


def collate_pad(batch):
    audio_list = list()
    melspec_list = list()
    for el in batch:
        audio_list.append(el[0])
        melspec_list.append(el[1].transpose(0, 1))
    return torch.nn.utils.rnn.pad_sequence(audio_list, batch_first=True, padding_value=0.0), \
           torch.nn.utils.rnn.pad_sequence(melspec_list, batch_first=True, padding_value=0.0).transpose(1, 2)


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
    collater = Collater()
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batchsize,
                              shuffle=True,
                              num_workers=4,
                              collate_fn=collate_pad)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=batchsize,
                              shuffle=False,
                              num_workers=4,
                              collate_fn=collate_pad)
    for epoch in range(epochs):
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        for datapoint in train_loader:
            batch_counter += 1
            # TODO check if this all works on a per batch basis or if it needs to be split up. Also check if collate needs to pad
            ############################
            #         Generator        #
            ############################
            gold_wave = datapoint[0].to(device)
            melspec = datapoint[1].to(device)
            pred_wave = g(melspec).squeeze(1)
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
    device = torch.device("cuda")
    train_dataset = MelGANDataset(list_of_paths=fl, type='train')
    valid_dataset = MelGANDataset(list_of_paths=fl, type='valid')
    generator = MelGANGenerator()
    multi_scale_discriminator = MelGANMultiScaleDiscriminator()
    if not os.path.exists("Models/MelGAN/SingleSpeaker/CSS10"):
        os.makedirs("Models/MelGAN/SingleSpeaker/CSS10")
    print("Training model")
    train_loop(batchsize=2,
               epochs=10,  # for testing
               generator=generator,
               discriminator=multi_scale_discriminator,
               train_dataset=train_dataset,
               valid_dataset=valid_dataset,
               device=device,
               generator_warmup_steps=10)  # for testing
