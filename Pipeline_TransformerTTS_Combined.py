"""
Train an autoregressive Transformer TTS model on the English single speaker dataset LJSpeech together
with the English multi speaker dataset LibriTTS
"""

import os
import random
import warnings

import torch
from torch.utils.data import ConcatDataset

from TransformerTTS.TransformerTTS import Transformer
from TransformerTTS.TransformerTTSDataset import TransformerTTSDataset
from TransformerTTS.transformer_tts_train_loop import train_loop
from Utility.path_to_transcript_dicts import build_path_to_transcript_dict_ljspeech, \
    build_path_to_transcript_dict_libritts

warnings.filterwarnings("ignore")

torch.manual_seed(13)
random.seed(13)

if __name__ == '__main__':
    print("Preparing")
    cache_dir_libritts = os.path.join("Corpora", "LibriTTS")
    cache_dir_ljspeech = os.path.join("Corpora", "LJSpeech", "with_spembs")
    if not os.path.exists(cache_dir_libritts):
        os.makedirs(cache_dir_libritts)
    if not os.path.exists(cache_dir_ljspeech):
        os.makedirs(cache_dir_ljspeech)

    save_dir = os.path.join("Models", "TransformerTTS", "MultiSpeaker", "Combined")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_set_libritts = TransformerTTSDataset(build_path_to_transcript_dict_libritts(),
                                               train=True,
                                               cache_dir=cache_dir_libritts,
                                               lang="en",
                                               min_len=10000,
                                               max_len=400000,
                                               spemb=True)
    valid_set_libritts = TransformerTTSDataset(build_path_to_transcript_dict_libritts(),
                                               train=False,
                                               cache_dir=cache_dir_libritts,
                                               lang="en",
                                               min_len=10000,
                                               max_len=400000,
                                               spemb=True)

    train_set_ljspeech = TransformerTTSDataset(build_path_to_transcript_dict_ljspeech(),
                                               train=True,
                                               cache_dir=cache_dir_ljspeech,
                                               lang="en",
                                               min_len=10000,
                                               max_len=400000,
                                               spemb=True)
    valid_set_ljspeech = TransformerTTSDataset(build_path_to_transcript_dict_ljspeech(),
                                               train=False,
                                               cache_dir=cache_dir_ljspeech,
                                               lang="en",
                                               min_len=10000,
                                               max_len=400000,
                                               spemb=True)

    train_set = ConcatDataset([train_set_libritts, train_set_ljspeech])
    valid_set = ConcatDataset([valid_set_libritts, valid_set_ljspeech])

    model = Transformer(idim=133, odim=80, spk_embed_dim=256)

    print("Training model")
    train_loop(net=model,
               train_dataset=train_set,
               valid_dataset=valid_set,
               device=torch.device("cuda:2"),
               config=model.get_conf(),
               save_directory=save_dir,
               epochs=300000,  # just kill the process at some point
               batchsize=64,
               gradient_accumulation=1,
               spemb=True,
               epochs_per_save=20,
               lang="en")
