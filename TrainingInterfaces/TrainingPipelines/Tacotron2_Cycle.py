import random

import torch
from torch.utils.data import ConcatDataset

from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.Tacotron2 import Tacotron2
from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.TacotronDataset import TacotronDataset
from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.tacotron2_train_loop import train_loop
from Utility.path_to_transcript_dicts import *


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume):
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
    cache_dir_hifitts = os.path.join("Corpora", "multispeaker_nvidia_hifitts")
    os.makedirs(cache_dir_hifitts, exist_ok=True)

    cache_dir_libri = os.path.join("Corpora", "LibriTTS")
    os.makedirs(cache_dir_libri, exist_ok=True)

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join("Models", "Tacotron2_MultispeakerEnglish")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    datasets = list()

    datasets.append(TacotronDataset(build_path_to_transcript_dict_nvidia_hifitts(),
                                    cache_dir=cache_dir_hifitts,
                                    lang="en",
                                    speaker_embedding=True,
                                    cut_silences=True,
                                    min_len_in_seconds=3,
                                    max_len_in_seconds=12,
                                    device=device))

    datasets.append(TacotronDataset(build_path_to_transcript_dict_libritts(),
                                    cache_dir=cache_dir_libri,
                                    lang="en",
                                    speaker_embedding=True,
                                    cut_silences=True,
                                    min_len_in_seconds=3,
                                    max_len_in_seconds=12,
                                    device=device))

    model = Tacotron2(idim=166, odim=80, spk_embed_dim=960)

    train_set = ConcatDataset(datasets)

    print("Training model")
    train_loop(net=model,
               train_dataset=train_set,
               device=device,
               save_directory=save_dir,
               steps=100000,
               batch_size=32,
               epochs_per_save=1,
               use_speaker_embedding=True,
               lang="en",
               lr=0.001,
               path_to_checkpoint="Models/Tacotron2_LibriTTS/best.pt",
               fine_tune=True,
               resume=resume,
               use_cycle_consistency_for_speakerembedding=True)
