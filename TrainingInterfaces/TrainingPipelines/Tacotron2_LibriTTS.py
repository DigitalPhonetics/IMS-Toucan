import os
import random

import torch

from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.Tacotron2 import Tacotron2
from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.TacotronDataset import TacotronDataset
from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.tacotron2_train_loop import train_loop
from Utility.path_to_transcript_dicts import build_path_to_transcript_dict_libritts as build_path_to_transcript_dict


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
    cache_dir = os.path.join("Corpora", "LibriTTS")
    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join("Models", "Tacotron2_LibriTTS")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path_to_transcript_dict = build_path_to_transcript_dict()

    train_set = TacotronDataset(path_to_transcript_dict,
                                cache_dir=cache_dir,
                                lang="en",
                                min_len_in_seconds=1,
                                max_len_in_seconds=10,
                                speaker_embedding=True)

    model = Tacotron2(idim=166, odim=80, spk_embed_dim=256, use_dtw_loss=True)

    print("Training model")
    train_loop(net=model,
               train_dataset=train_set,
               device=device,
               save_directory=save_dir,
               steps=500000,  # this has a lot more data than the others, so it can learn for longer
               batch_size=84,
               epochs_per_save=5,  # more datapoints per epoch needs fewer epochs per save
               use_speaker_embedding=True,
               lang="en",
               lr=0.05,
               warmup_steps=8000,
               path_to_checkpoint=resume_checkpoint,
               fine_tune=finetune)
