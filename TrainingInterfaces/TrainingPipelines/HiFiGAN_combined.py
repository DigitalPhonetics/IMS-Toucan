import random

import torch

from TrainingInterfaces.Spectrogram_to_Wave.HiFIGAN.HiFiGAN import HiFiGANGenerator
from TrainingInterfaces.Spectrogram_to_Wave.HiFIGAN.HiFiGAN import HiFiGANMultiScaleMultiPeriodDiscriminator
from TrainingInterfaces.Spectrogram_to_Wave.HiFIGAN.HiFiGANDataset import HiFiGANDataset
from TrainingInterfaces.Spectrogram_to_Wave.HiFIGAN.hifigan_train_loop import train_loop
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

    giant_file_list = list()
    giant_file_list += get_file_list_elizabeth()
    giant_file_list += get_file_list_libritts()
    giant_file_list += get_file_list_thorsten()
    giant_file_list += get_file_list_eva()
    giant_file_list += get_file_list_ljspeech()
    giant_file_list += get_file_list_css10ch()
    giant_file_list += get_file_list_css10du()
    giant_file_list += get_file_list_css10es()
    giant_file_list += get_file_list_css10fi()
    giant_file_list += get_file_list_css10fr()
    giant_file_list += get_file_list_css10ge()
    giant_file_list += get_file_list_css10gr()
    giant_file_list += get_file_list_css10hu()
    giant_file_list += get_file_list_css10jp()
    giant_file_list += get_file_list_css10ru()
    giant_file_list += get_file_list_hokuspokus()
    giant_file_list += get_file_list_karlsson()
    giant_file_list += get_file_list_nancy()

    train_set = HiFiGANDataset(list_of_paths=giant_file_list)
    generator = HiFiGANGenerator()
    generator.reset_parameters()
    multi_scale_discriminator = HiFiGANMultiScaleMultiPeriodDiscriminator()

    print("Training model")
    train_loop(batch_size=16,
               steps=2000000,
               generator=generator,
               discriminator=multi_scale_discriminator,
               train_dataset=train_set,
               device=device,
               model_save_dir=model_save_dir,
               path_to_checkpoint=resume_checkpoint)
