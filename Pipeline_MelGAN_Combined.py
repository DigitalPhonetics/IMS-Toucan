"""
Train non-autoregressive spectrogram inversion model on a combination of multiple large datasets

Spectrogram inversion is language and speaker independent,
so throwing together all datasets gives the best results.

"""

import gc
import os
import random
import warnings

import torch
from torch.utils.data import ConcatDataset

from MelGAN.MelGANDataset import MelGANDataset
from MelGAN.MelGANGenerator import MelGANGenerator
from MelGAN.MelGANMultiScaleDiscriminator import MelGANMultiScaleDiscriminator
from MelGAN.melgan_train_loop import train_loop
from Utility.file_lists import get_file_list_css10de, get_file_list_libritts, get_file_list_ljspeech

warnings.filterwarnings("ignore")

torch.manual_seed(13)
random.seed(13)

if __name__ == '__main__':
    print("Preparing")
    model_save_dir = "Models/MelGAN/MultiSpeaker/Combined"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    train_set_libri = MelGANDataset(list_of_paths=get_file_list_libritts()[:-300])
    valid_set_libri = MelGANDataset(list_of_paths=get_file_list_libritts()[-300:])
    train_set_lj = MelGANDataset(list_of_paths=get_file_list_ljspeech()[:-100])
    valid_set_lj = MelGANDataset(list_of_paths=get_file_list_ljspeech()[-100:])
    train_set_css10de = MelGANDataset(list_of_paths=get_file_list_css10de()[:-100])
    valid_set_css10de = MelGANDataset(list_of_paths=get_file_list_css10de()[-100:])

    train_set = ConcatDataset([train_set_libri, train_set_lj, train_set_css10de])
    valid_set = ConcatDataset([valid_set_libri, valid_set_lj, valid_set_css10de])

    gc.collect()

    generator = MelGANGenerator()
    generator.reset_parameters()
    multi_scale_discriminator = MelGANMultiScaleDiscriminator()

    print("Training model")
    train_loop(batchsize=32,
               epochs=6000000,  # just kill the process at some point
               generator=generator,
               discriminator=multi_scale_discriminator,
               train_dataset=train_set,
               valid_dataset=valid_set,
               device=torch.device("cuda:2"),
               generator_warmup_steps=100000,
               model_save_dir=model_save_dir)
