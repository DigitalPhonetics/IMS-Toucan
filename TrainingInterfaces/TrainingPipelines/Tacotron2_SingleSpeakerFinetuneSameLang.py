import os
import random

import soundfile
import torch

from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.Tacotron2 import Tacotron2
from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.TacotronDataset import TacotronDataset
from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.tacotron2_train_loop import train_loop
from Utility.file_lists import get_file_list_ljspeech
from Utility.path_to_transcript_dicts import build_path_to_transcript_dict_ljspeech as build_path_to_transcript_dict


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
    cache_dir = os.path.join("Corpora", "LJSpeech_low_res")
    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join("Models", "Tacotron2_SameLangFinetune")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path_to_transcript_dict_ = build_path_to_transcript_dict()
    path_to_transcript_dict = dict()

    paths = get_file_list_ljspeech()
    used_samples = set()
    total_len = 0.0
    while total_len < 30.0 * 60.0:
        path = random.choice(paths)
        x, sr = soundfile.read(path)
        duration = len(x) / sr
        if 10 > duration > 5 and path not in used_samples:
            used_samples.add(path)
            total_len += duration

    print(f"Collected {total_len / 60.0} minutes worth of samples.")

    for key in path_to_transcript_dict_:
        if key in used_samples:
            path_to_transcript_dict[key] = path_to_transcript_dict_[key]

    train_set = TacotronDataset(path_to_transcript_dict,
                                cache_dir=cache_dir,
                                lang="en")

    model = Tacotron2()

    print("Training model")
    train_loop(net=model,
               train_dataset=train_set,
               device=device,
               save_directory=save_dir,
               steps=6000,
               batch_size=32,
               epochs_per_save=20,
               lang="en",
               lr=0.001,
               path_to_checkpoint="Models/Tacotron2_Nancy/best.pt",
               fine_tune=True,
               resume=resume,
               cycle_loss_start_steps=2000)
