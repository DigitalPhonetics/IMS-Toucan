import random

import torch

from TrainingInterfaces.Spectrogram_to_Wave.MelGAN.MelGANDataset import MelGANDataset
from TrainingInterfaces.Spectrogram_to_Wave.MelGAN.MelGANGenerator import MelGANGenerator
from TrainingInterfaces.Spectrogram_to_Wave.MelGAN.MelGANMultiScaleDiscriminator import MelGANMultiScaleDiscriminator
from TrainingInterfaces.Spectrogram_to_Wave.MelGAN.melgan_train_loop import train_loop
from Utility.file_lists import *


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
        model_save_dir = "Models/MelGAN_combined"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    file_lists = list()
    file_lists.append(get_file_list_elizabeth())
    file_lists.append(get_file_list_libritts())
    file_lists.append(get_file_list_thorsten())
    file_lists.append(get_file_list_eva())
    file_lists.append(get_file_list_ljspeech())
    file_lists.append(get_file_list_css10ch())
    file_lists.append(get_file_list_css10du())
    file_lists.append(get_file_list_css10es())
    file_lists.append(get_file_list_css10fi())
    file_lists.append(get_file_list_css10fr())
    file_lists.append(get_file_list_css10ge())
    file_lists.append(get_file_list_css10gr())
    file_lists.append(get_file_list_css10hu())
    file_lists.append(get_file_list_css10jp())
    file_lists.append(get_file_list_css10ru())
    file_lists.append(get_file_list_hokuspokus())
    file_lists.append(get_file_list_karlsson())
    file_lists.append(get_file_list_nancy())

    datasets = list()

    for file_list in file_lists:
        datasets.append(HiFiGANDataset(list_of_paths=file_list))
    train_set = torch.utils.data.ConcatDataset(datasets)
    
    generator = MelGANGenerator()
    generator.reset_parameters()
    multi_scale_discriminator = MelGANMultiScaleDiscriminator()

    print("Training model")
    train_loop(batch_size=16,
               steps=2000000,
               generator=generator,
               discriminator=multi_scale_discriminator,
               train_dataset=train_set,
               device=device,
               generator_warmup_steps=100000,
               model_save_dir=model_save_dir,
               path_to_checkpoint=resume_checkpoint)
