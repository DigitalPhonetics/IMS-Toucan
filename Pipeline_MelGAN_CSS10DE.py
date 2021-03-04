"""
Train non-autoregressive spectrogram inversion model on the german single speaker dataset by Hokuspokus
"""

import os
import random
import warnings

import torch

from MelGAN.MelGANDataset import MelGANDataset
from MelGAN.MelGANGenerator import MelGANGenerator
from MelGAN.MelGANMultiScaleDiscriminator import MelGANMultiScaleDiscriminator
from MelGAN.melgan_train_loop import train_loop

warnings.filterwarnings("ignore")

torch.manual_seed(17)
random.seed(17)


def get_file_list():
    file_list = list()
    with open("Corpora/CSS10_DE/transcript.txt", encoding="utf8") as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            file_list.append("Corpora/CSS10_DE/" + line.split("|")[0])
    return file_list


if __name__ == '__main__':
    print("Preparing")
    fl = get_file_list()
    device = torch.device("cuda")
    train_dataset = MelGANDataset(list_of_paths=fl[:-100])
    valid_dataset = MelGANDataset(list_of_paths=fl[-100:])
    generator = MelGANGenerator()
    generator.reset_parameters()
    multi_scale_discriminator = MelGANMultiScaleDiscriminator()
    if not os.path.exists("Models/MelGAN/SingleSpeaker/CSS10DE"):
        os.makedirs("Models/MelGAN/SingleSpeaker/CSS10DE")
    print("Training model")
    train_loop(batchsize=32,
               epochs=60000,  # just kill the process at some point
               generator=generator,
               discriminator=multi_scale_discriminator,
               train_dataset=train_dataset,
               valid_dataset=valid_dataset,
               device=device,
               generator_warmup_steps=200000,
               model_save_dir="Models/MelGAN/SingleSpeaker/CSS10DE")
