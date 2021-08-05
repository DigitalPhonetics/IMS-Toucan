import os
import random

import torch

from TrainingInterfaces.Spectrogram_to_Wave.MelGAN.MelGANDataset import MelGANDataset
from TrainingInterfaces.Spectrogram_to_Wave.MelGAN.MelGANGenerator import MelGANGenerator
from TrainingInterfaces.Spectrogram_to_Wave.MelGAN.MelGANMultiScaleDiscriminator import MelGANMultiScaleDiscriminator
from TrainingInterfaces.Spectrogram_to_Wave.MelGAN.melgan_train_loop import train_loop
from Utility.file_lists import get_file_list_ljspeech as get_file_list


def run(gpu_id, resume_checkpoint, finetune, model_dir):
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
    if model_dir is not None:
        model_save_dir = model_dir
    else:
        model_save_dir = "Models/MelGAN_LJSpeech"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    train_set = MelGANDataset(list_of_paths=get_file_list())
    generator = MelGANGenerator()
    generator.reset_parameters()
    multi_scale_discriminator = MelGANMultiScaleDiscriminator()

    print("Training model")
    train_loop(batch_size=16,
               steps=500000,
               generator=generator,
               discriminator=multi_scale_discriminator,
               train_dataset=train_set,
               device=device,
               generator_warmup_steps=100000,
               model_save_dir=model_save_dir,
               path_to_checkpoint=resume_checkpoint)
