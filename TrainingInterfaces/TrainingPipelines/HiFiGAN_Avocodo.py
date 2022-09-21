import os.path
import time

import torch
import wandb
from torch.utils.data import ConcatDataset

from TrainingInterfaces.Spectrogram_to_Wave.HiFiGAN.HiFiGAN import AvocodoHiFiGANJointDiscriminator
from TrainingInterfaces.Spectrogram_to_Wave.HiFiGAN.HiFiGAN import HiFiGANGenerator
from TrainingInterfaces.Spectrogram_to_Wave.HiFiGAN.HiFiGANDataset import HiFiGANDataset
from TrainingInterfaces.Spectrogram_to_Wave.HiFiGAN.hifigan_train_loop import train_loop
from Utility.path_to_transcript_dicts import *


def run(gpu_id, resume_checkpoint, finetune, resume, model_dir, use_wandb):
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
        model_save_dir = "Models/HiFiGAN_Avocodo"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # sampling multiple times from the dataset, because it's too big to fit all at once
    for run_id in range(1000):

        file_lists = list()
        file_lists.append(random.sample(list(build_path_to_transcript_dict_mls_italian().keys()), 1500))
        file_lists.append(random.sample(list(build_path_to_transcript_dict_mls_french().keys()), 1500))
        file_lists.append(random.sample(list(build_path_to_transcript_dict_mls_dutch().keys()), 1500))
        file_lists.append(random.sample(list(build_path_to_transcript_dict_mls_polish().keys()), 1500))
        file_lists.append(random.sample(list(build_path_to_transcript_dict_mls_spanish().keys()), 1500))
        file_lists.append(random.sample(list(build_path_to_transcript_dict_mls_portuguese().keys()), 1500))
        file_lists.append(random.sample(list(build_path_to_transcript_dict_karlsson().keys()), 500))
        file_lists.append(random.sample(list(build_path_to_transcript_dict_eva().keys()), 500))
        file_lists.append(random.sample(list(build_path_to_transcript_dict_bernd().keys()), 500))
        file_lists.append(random.sample(list(build_path_to_transcript_dict_friedrich().keys()), 500))
        file_lists.append(random.sample(list(build_path_to_transcript_dict_hokus().keys()), 500))
        file_lists.append(random.sample(list(build_path_to_transcript_dict_hui_others().keys()), 1500))
        file_lists.append(random.sample(list(build_path_to_transcript_dict_elizabeth().keys()), 500))
        file_lists.append(random.sample(list(build_path_to_transcript_dict_nancy().keys()), 500))
        file_lists.append(random.sample(list(build_path_to_transcript_dict_hokuspokus().keys()), 500))
        file_lists.append(random.sample(list(build_path_to_transcript_dict_fluxsing().keys()), 500))
        file_lists.append(random.sample(list(build_path_to_transcript_dict_vctk().keys()), 500))
        file_lists.append(random.sample(list(build_path_to_transcript_dict_libritts_all_clean().keys()), 3000))
        file_lists.append(random.sample(list(build_path_to_transcript_dict_ljspeech().keys()), 500))
        file_lists.append(random.sample(list(build_path_to_transcript_dict_css10cmn().keys()), 500))
        file_lists.append(random.sample(list(build_path_to_transcript_dict_vietTTS().keys()), 500))
        file_lists.append(random.sample(list(build_path_to_transcript_dict_thorsten().keys()), 500))
        file_lists.append(random.sample(list(build_path_to_transcript_dict_css10el().keys()), 500))
        file_lists.append(random.sample(list(build_path_to_transcript_dict_css10nl().keys()), 500))
        file_lists.append(random.sample(list(build_path_to_transcript_dict_css10fi().keys()), 500))
        file_lists.append(random.sample(list(build_path_to_transcript_dict_css10ru().keys()), 500))
        file_lists.append(random.sample(list(build_path_to_transcript_dict_css10hu().keys()), 500))
        file_lists.append(random.sample(list(build_path_to_transcript_dict_css10es().keys()), 500))
        file_lists.append(random.sample(list(build_path_to_transcript_dict_css10fr().keys()), 500))
        file_lists.append(random.sample(list(build_path_to_transcript_dict_nvidia_hifitts().keys()), 500))
        file_lists.append(random.sample(list(build_path_to_transcript_dict_spanish_blizzard_train().keys()), 500))
        file_lists.append(random.sample(list(build_path_to_transcript_dict_aishell3().keys()), 1500))
        file_lists.append(random.sample(list(build_path_to_transcript_dict_VIVOS_viet().keys()), 500))
        file_lists.append(random.sample(list(build_path_to_transcript_dict_RAVDESS().keys()), 500))
        file_lists.append(random.sample(list(build_path_to_transcript_dict_ESDS().keys()), 1500))

        datasets = list()

        for index, file_list in enumerate(file_lists):
            datasets.append(HiFiGANDataset(list_of_paths=file_list, cache_dir=f"Corpora/{index}", use_random_corruption=False))
        train_set = ConcatDataset(datasets)

        generator = HiFiGANGenerator()
        generator.reset_parameters()
        discriminator = AvocodoHiFiGANJointDiscriminator()

        print("Training model")
        if use_wandb:
            wandb.init(name=f"{__name__.split('.')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}")
            wandb.watch(generator, log_graph=True)
        if run_id == 0:
            train_loop(batch_size=32,
                       epochs=20,
                       generator=generator,
                       discriminator=discriminator,
                       train_dataset=train_set,
                       device=device,
                       epochs_per_save=2,
                       model_save_dir=model_save_dir,
                       path_to_checkpoint=resume_checkpoint,
                       resume=resume,
                       use_signal_processing_losses=False,
                       use_wandb=use_wandb)
        else:
            train_loop(batch_size=32,
                       epochs=20,
                       generator=generator,
                       discriminator=discriminator,
                       train_dataset=train_set,
                       device=device,
                       epochs_per_save=2,
                       model_save_dir=model_save_dir,
                       path_to_checkpoint=None,
                       resume=True,
                       use_signal_processing_losses=False,
                       use_wandb=use_wandb)
        if use_wandb:
            wandb.finish()
