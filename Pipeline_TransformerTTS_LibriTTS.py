"""
Train an autoregressive Transformer TTS model on the English single speaker dataset LJSpeech
"""

import os
import random
import warnings

import torch

from TransformerTTS.TransformerTTS import Transformer
from TransformerTTS.TransformerTTSDataset import TransformerTTSDataset
from TransformerTTS.transformer_tts_train_loop import train_loop
from Utility.path_to_transcript_dicts import build_path_to_transcript_dict_libritts

warnings.filterwarnings("ignore")

torch.manual_seed(13)
random.seed(13)

if __name__ == '__main__':
    print("Preparing")
    cache_dir = os.path.join("Corpora", "LibriTTS")
    save_dir = os.path.join("Models", "TransformerTTS", "MultiSpeaker", "LibriTTS")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path_to_transcript_dict = build_path_to_transcript_dict_libritts()

    train_set = TransformerTTSDataset(path_to_transcript_dict,
                                      train=True,
                                      cache_dir=cache_dir,
                                      lang="en",
                                      min_len=10000,
                                      max_len=400000,
                                      spemb=True)
    valid_set = TransformerTTSDataset(path_to_transcript_dict,
                                      train=False,
                                      cache_dir=cache_dir,
                                      lang="en",
                                      min_len=10000,
                                      max_len=400000,
                                      spemb=True)

    model = Transformer(idim=131, odim=80, spk_embed_dim=256, reduction_factor=5)

    print("Training model")
    train_loop(net=model,
               train_dataset=train_set,
               valid_dataset=valid_set,
               device=torch.device("cuda:3"),
               config=model.get_conf(),
               save_directory=save_dir,
               epochs=300000,  # just kill the process at some point
               batchsize=16,
               gradient_accumulation=4,
               spemb=True)
