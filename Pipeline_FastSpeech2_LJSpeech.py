"""
Train a non-autoregressive FastSpeech 2 model on the English single speaker dataset LJSpeech

This requires having a trained TransformerTTS model in the right directory to knowledge distill the durations.
"""

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import random
import warnings

import torch

from FastSpeech2.FastSpeech2 import FastSpeech2
from FastSpeech2.FastSpeechDataset import FastSpeechDataset
from FastSpeech2.fastspeech2_train_loop import train_loop
from Utility.path_to_transcript_dicts import build_path_to_transcript_dict_ljspeech

warnings.filterwarnings("ignore")

torch.manual_seed(13)
random.seed(13)

if __name__ == '__main__':
    print("Preparing")
    cache_dir = os.path.join("Corpora", "LJSpeech")
    save_dir = os.path.join("Models", "FastSpeech2", "SingleSpeaker", "LJSpeech")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path_to_transcript_dict = build_path_to_transcript_dict_ljspeech()

    device = torch.device("cuda")

    train_set = FastSpeechDataset(path_to_transcript_dict,
                                  train=True,
                                  acoustic_model_name="Transformer_English_Single.pt",
                                  cache_dir=cache_dir,
                                  lang="en",
                                  min_len_in_seconds=1,
                                  max_len_in_seconds=10,
                                  device=device)
    valid_set = FastSpeechDataset(path_to_transcript_dict,
                                  train=False,
                                  acoustic_model_name="Transformer_English_Single.pt",
                                  cache_dir=cache_dir,
                                  lang="en",
                                  min_len_in_seconds=1,
                                  max_len_in_seconds=10,
                                  device=device)

    for index in range(len(train_set)):
        print(len(train_set[index][4]))
        print(len(train_set[index][5]))
        print(len(train_set[index][6]))
        print("\n\n\n")

    model = FastSpeech2(idim=133, odim=80, spk_embed_dim=None)

    print("Training model")
    train_loop(net=model,
               train_dataset=train_set,
               valid_dataset=valid_set,
               device=device,
               config=model.get_conf(),
               save_directory=save_dir,
               epochs=300000,  # just kill the process at some point
               batchsize=32,
               gradient_accumulation=1)
