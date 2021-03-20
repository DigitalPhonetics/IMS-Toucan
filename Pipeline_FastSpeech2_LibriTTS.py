"""
Train a non-autoregressive FastSpeech 2 model on the german single speaker dataset by Hokuspokus

This requires having a trained TransformerTTS model in the right directory to knowledge distill the durations.
"""

import os
import random
import warnings

import torch

from FastSpeech2.FastSpeech2 import FastSpeech2
from FastSpeech2.FastSpeechDataset import FastSpeechDataset
from FastSpeech2.fastspeech2_train_loop import train_loop
from Utility.path_to_transcript_dicts import build_path_to_transcript_dict_libritts

warnings.filterwarnings("ignore")

torch.manual_seed(17)
random.seed(17)

if __name__ == '__main__':
    print("Preparing")
    cache_dir = os.path.join("Corpora", "LibriTTS")
    save_dir = os.path.join("Models", "FastSpeech2", "MultiSpeaker", "LibriTTS")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path_to_transcript_dict = build_path_to_transcript_dict_libritts()

    train_set = FastSpeechDataset(path_to_transcript_dict,
                                  train=True,
                                  acoustic_model_name="Transformer_English_Multi.pt",
                                  cache_dir=cache_dir,
                                  lang="en",
                                  min_len=0,
                                  max_len=1000000,
                                  spemb=True)
    valid_set = FastSpeechDataset(path_to_transcript_dict,
                                  train=False,
                                  acoustic_model_name="Transformer_English_Multi.pt",
                                  cache_dir=cache_dir,
                                  lang="en",
                                  min_len=0,
                                  max_len=1000000,
                                  spemb=True)

    model = FastSpeech2(idim=131, odim=80, spk_embed_dim=256)

    print("Training model")
    train_loop(net=model,
               train_dataset=train_set,
               valid_dataset=valid_set,
               device=torch.device("cpu"),
               config=model.get_conf(),
               save_directory=save_dir,
               epochs=300000,  # just kill the process at some point
               batchsize=32,
               gradient_accumulation=1,
               spemb=True)
