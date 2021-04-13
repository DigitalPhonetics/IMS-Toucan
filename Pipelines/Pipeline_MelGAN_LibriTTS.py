import os
import random

import torch

from MelGAN.MelGANDataset import MelGANDataset
from MelGAN.MelGANGenerator import MelGANGenerator
from MelGAN.MelGANMultiScaleDiscriminator import MelGANMultiScaleDiscriminator
from MelGAN.melgan_train_loop import train_loop
from Utility.file_lists import get_file_list_libritts


def run(gpu_id, resume_checkpoint, finetune):
    if gpu_id == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")

    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
        device = torch.device("cuda")

    torch.manual_seed(13)
    random.seed(13)

    print("Preparing")
    model_save_dir = "Models/MelGAN_LibriTTS"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    melgan_cache_dir = "Corpora/MelGAN"
    if not os.path.exists(melgan_cache_dir):
        os.makedirs(melgan_cache_dir)

    train_set_libri = MelGANDataset(list_of_paths=get_file_list_libritts()[:-50],
                                    cache=os.path.join(melgan_cache_dir, "LibriTTS_train.txt"))
    valid_set_libri = MelGANDataset(list_of_paths=get_file_list_libritts()[-50:],
                                    cache=os.path.join(melgan_cache_dir, "LibriTTS_valid.txt"))

    generator = MelGANGenerator()
    generator.reset_parameters()
    multi_scale_discriminator = MelGANMultiScaleDiscriminator()

    print("Training model")
    train_loop(batch_size=16,
               steps=2000000,
               generator=generator,
               discriminator=multi_scale_discriminator,
               train_dataset=train_set_libri,
               valid_dataset=valid_set_libri,
               device=device,
               generator_warmup_steps=100000,
               model_save_dir=model_save_dir,
               path_to_checkpoint=resume_checkpoint)
