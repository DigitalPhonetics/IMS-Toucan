import random

import torch
from torch.utils.data import ConcatDataset

from TrainingInterfaces.Spectrogram_to_Wave.HiFIGAN.HiFiGAN import HiFiGANGenerator
from TrainingInterfaces.Spectrogram_to_Wave.HiFIGAN.HiFiGAN import HiFiGANMultiScaleMultiPeriodDiscriminator
from TrainingInterfaces.Spectrogram_to_Wave.HiFIGAN.HiFiGANDataset import HiFiGANDataset
from TrainingInterfaces.Spectrogram_to_Wave.HiFIGAN.hifigan_train_loop import train_loop
from Utility.file_lists import *


def run(gpu_id, resume_checkpoint, finetune, resume, model_dir):
    if gpu_id == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")

    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
        device = torch.device("cuda")

    torch.manual_seed(131714)
    random.seed(131714)
    torch.random.manual_seed(131714)

    print("Preparing")
    if model_dir is not None:
        model_save_dir = model_dir
    else:
        model_save_dir = "Models/HiFiGAN_combined"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    file_lists = list()
    file_lists.append(get_file_list_css10gr()[:200])
    file_lists.append(get_file_list_elizabeth()[:200])
    file_lists.append(random.sample(get_file_list_libritts(), 200))
    file_lists.append(get_file_list_thorsten()[:200])
    file_lists.append(get_file_list_eva()[:200])
    file_lists.append(get_file_list_ljspeech()[:200])
    file_lists.append(get_file_list_css10ch()[:200])
    file_lists.append(get_file_list_css10du()[:200])
    file_lists.append(get_file_list_css10es()[:200])
    file_lists.append(get_file_list_css10fi()[:200])
    file_lists.append(get_file_list_css10fr()[:200])
    file_lists.append(get_file_list_css10de()[:200])
    file_lists.append(get_file_list_css10hu()[:200])
    file_lists.append(get_file_list_css10jp()[:200])
    file_lists.append(get_file_list_css10ru()[:200])
    file_lists.append(get_file_list_karlsson())
    file_lists.append(get_file_list_nancy())
    file_lists.append(random.sample(get_file_list_nvidia_hifitts(), 200))

    datasets = list()

    for index, file_list in enumerate(file_lists):
        datasets.append(HiFiGANDataset(list_of_paths=file_list, cache_dir=f"Corpora/{index}"))
    train_set = ConcatDataset(datasets)

    generator = HiFiGANGenerator()
    generator.reset_parameters()
    discriminator = HiFiGANMultiScaleMultiPeriodDiscriminator()

    print("Training model")
    train_loop(batch_size=16,
               steps=2500000,
               generator=generator,
               discriminator=discriminator,
               train_dataset=train_set,
               device=device,
               epochs_per_save=1,
               model_save_dir=model_save_dir,
               path_to_checkpoint=resume_checkpoint)
