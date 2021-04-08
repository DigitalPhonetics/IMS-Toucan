"""
Train a non-autoregressive FastSpeech 2 model on the English multi speaker dataset LibriTTS

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
from Utility.path_to_transcript_dicts import build_path_to_transcript_dict_libritts

warnings.filterwarnings("ignore")

torch.manual_seed(13)
random.seed(13)

if __name__ == '__main__':
    print("Preparing")
    cache_dir = os.path.join("Corpora", "LibriTTS")
    save_dir = os.path.join("Models", "FastSpeech2_LibriTTS")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path_to_transcript_dict = build_path_to_transcript_dict_libritts()

    device = torch.device("cuda")

    train_set = FastSpeechDataset(path_to_transcript_dict, train=True, acoustic_model_name="TransformerTTS_LibriTTS/best.pt", cache_dir=cache_dir, lang="en",
                                  min_len_in_seconds=1, max_len_in_seconds=10, device=device, spemb=True)
    valid_set = FastSpeechDataset(path_to_transcript_dict, train=False, acoustic_model_name="TransformerTTS_LibriTTS/best.pt", cache_dir=cache_dir, lang="en",
                                  min_len_in_seconds=1, max_len_in_seconds=10, device=device, spemb=True)

    model = FastSpeech2(idim=133, odim=80, spk_embed_dim=256)

    print("Training model")
    train_loop(net=model, train_dataset=train_set, valid_dataset=valid_set, device=device, save_directory=save_dir, steps=400000, batch_size=32,
               gradient_accumulation=1, epochs_per_save=10, use_speaker_embedding=True, lang="en", lr=0.05, warmup_steps=8000)
