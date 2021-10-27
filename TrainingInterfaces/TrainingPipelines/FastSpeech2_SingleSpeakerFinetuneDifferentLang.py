import os
import random

import soundfile
import torch

from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeechDataset import FastSpeechDataset
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.fastspeech2_train_loop import train_loop
from Utility.file_lists import get_file_list_css10de
from Utility.path_to_transcript_dicts import build_path_to_transcript_dict_css10de as build_path_to_transcript_dict


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
    cache_dir = os.path.join("Corpora", "Hokuspokus_low_res")
    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join("Models", "FastSpeech2_DiffLangLangFinetune")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path_to_transcript_dict_ = build_path_to_transcript_dict()
    path_to_transcript_dict = dict()

    paths = get_file_list_css10de()
    used_samples = set()
    total_len = 0.0
    while total_len < 30.0 * 60.0:
        path = random.choice(paths)
        if "meisterfloh" in path:  # in that book she uses the nicest microphone
            x, sr = soundfile.read(path)
            duration = len(x) / sr
            if 10 > duration > 5 and path not in used_samples:
                used_samples.add(path)
                total_len += duration

    print(f"Collected {total_len / 60.0} minutes worth of samples.")

    for key in path_to_transcript_dict_:
        if key in used_samples:
            path_to_transcript_dict[key] = path_to_transcript_dict_[key]

    acoustic_checkpoint_path = os.path.join("Models", "Tacotron2_HokusPokus_Aligner", "best.pt")

    train_set = FastSpeechDataset(path_to_transcript_dict,
                                  cache_dir=cache_dir,
                                  acoustic_checkpoint_path=acoustic_checkpoint_path,
                                  lang="de",
                                  device=device)

    model = FastSpeech2()

    print("Training model")
    train_loop(net=model,
               train_dataset=train_set,
               device=device,
               save_directory=save_dir,
               steps=30000,
               batch_size=20,
               lang="de",
               lr=0.001,
               warmup_steps=14000,
               path_to_checkpoint="Models/Singe_Step_LAML_FastSpeech2/best.pt",
               fine_tune=True,
               resume=resume)
