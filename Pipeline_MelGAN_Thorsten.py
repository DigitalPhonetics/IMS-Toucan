"""
Train non-autoregressive spectrogram inversion model
"""

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import warnings

import torch

from MelGAN.MelGANDataset import MelGANDataset
from MelGAN.MelGANGenerator import MelGANGenerator
from MelGAN.MelGANMultiScaleDiscriminator import MelGANMultiScaleDiscriminator
from MelGAN.melgan_train_loop import train_loop
from Utility.file_lists import get_file_list_thorsten

warnings.filterwarnings("ignore")

torch.manual_seed(13)
random.seed(13)

if __name__ == '__main__':
    print("Preparing")
    model_save_dir = "Models/MelGAN_Thorsten"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    melgan_cache_dir = "Corpora/MelGAN"
    if not os.path.exists(melgan_cache_dir):
        os.makedirs(melgan_cache_dir)

    train_set_thorsten = MelGANDataset(list_of_paths=get_file_list_thorsten()[:-50],
                                       cache=os.path.join(melgan_cache_dir, "thorsten_train.txt"))
    valid_set_thorsten = MelGANDataset(list_of_paths=get_file_list_thorsten()[-50:],
                                       cache=os.path.join(melgan_cache_dir, "thorsten_valid.txt"))

    generator = MelGANGenerator()
    generator.reset_parameters()
    multi_scale_discriminator = MelGANMultiScaleDiscriminator()

    print("Training model")
    train_loop(batch_size=16,
               steps=2000000,
               generator=generator,
               discriminator=multi_scale_discriminator,
               train_dataset=train_set_thorsten,
               valid_dataset=valid_set_thorsten,
               device=torch.device("cuda"),
               generator_warmup_steps=100000,
               model_save_dir=model_save_dir)
