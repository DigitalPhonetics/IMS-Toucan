"""
Train an autoregressive Transformer TTS model on the German single speaker dataset by Hokuspokus
"""
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import warnings

import torch

from TransformerTTS.TransformerTTS import Transformer
from TransformerTTS.TransformerTTSDataset import TransformerTTSDataset
from TransformerTTS.transformer_tts_train_loop import train_loop

warnings.filterwarnings("ignore")
from Utility.path_to_transcript_dicts import build_path_to_transcript_dict_hokuspokus

torch.manual_seed(13)
random.seed(13)

if __name__ == '__main__':
    print("Preparing")
    cache_dir = os.path.join("Corpora", "Hokuspokus")
    save_dir = os.path.join("Models", "TransformerTTS", "SingleSpeaker", "Hokuspokus")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path_to_transcript_dict = build_path_to_transcript_dict_hokuspokus()

    train_set = TransformerTTSDataset(path_to_transcript_dict,
                                      train=True,
                                      cache_dir=cache_dir,
                                      lang="de",
                                      min_len=0,
                                      max_len=1000000,
                                      rebuild_cache=False)
    valid_set = TransformerTTSDataset(path_to_transcript_dict,
                                      train=False,
                                      cache_dir=cache_dir,
                                      lang="de",
                                      min_len=0,
                                      max_len=1000000,
                                      rebuild_cache=False)

    model = Transformer(idim=134, odim=80, spk_embed_dim=None, reduction_factor=1)

    print("Training model")
    train_loop(net=model,
               train_dataset=train_set,
               valid_dataset=valid_set,
               device=torch.device("cuda"),
               config=model.get_conf(),
               save_directory=save_dir,
               epochs=300000,  # just kill the process at some point
               batchsize=8,
               gradient_accumulation=8,
               epochs_per_save=10,
               spemb=False,
               lang="de",
               lr=0.001,
               warmup_steps=14000)
