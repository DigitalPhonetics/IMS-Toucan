"""
Train an autoregressive Transformer TTS model on the English single speaker dataset LJSpeech
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
from Utility.path_to_transcript_dicts import build_path_to_transcript_dict_ljspeech

warnings.filterwarnings("ignore")

torch.manual_seed(13)
random.seed(13)

if __name__ == '__main__':
    print("Preparing")
    cache_dir = os.path.join("Corpora", "LJSpeech")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    path_to_transcript_dict = build_path_to_transcript_dict_ljspeech()

    train_set = TransformerTTSDataset(path_to_transcript_dict,
                                      train=True,
                                      cache_dir=cache_dir,
                                      lang="en",
                                      min_len_in_seconds=1,
                                      max_len_in_seconds=17,
                                      rebuild_cache=False)
    valid_set = TransformerTTSDataset(path_to_transcript_dict,
                                      train=False,
                                      cache_dir=cache_dir,
                                      lang="en",
                                      min_len_in_seconds=1,
                                      max_len_in_seconds=17,
                                      rebuild_cache=False)

    #######################################################################################
    save_dir = os.path.join("Models", "TransformerTTS", "SingleSpeaker", "lr0005")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model = Transformer(idim=134, odim=80, spk_embed_dim=None)
    train_loop(net=model,
               train_dataset=train_set,
               valid_dataset=valid_set,
               device=torch.device("cuda"),
               config=model.get_conf(),
               save_directory=save_dir,
               epochs=100,
               batchsize=14,
               gradient_accumulation=5,
               epochs_per_save=10,
               spemb=False,
               lang="en",
               lr=0.0005,
               warmup_steps=8000,
               checkpoint="checkpoint_2035.pt")

    ############################################################################
    save_dir = os.path.join("Models", "TransformerTTS", "SingleSpeaker", "lr0001")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model = Transformer(idim=134, odim=80, spk_embed_dim=None)
    train_loop(net=model,
               train_dataset=train_set,
               valid_dataset=valid_set,
               device=torch.device("cuda"),
               config=model.get_conf(),
               save_directory=save_dir,
               epochs=100,
               batchsize=14,
               gradient_accumulation=5,
               epochs_per_save=10,
               spemb=False,
               lang="en",
               lr=0.0001,
               warmup_steps=8000)

    #######################################################################################
    save_dir = os.path.join("Models", "TransformerTTS", "SingleSpeaker", "lr005")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model = Transformer(idim=134, odim=80, spk_embed_dim=None)
    train_loop(net=model,
               train_dataset=train_set,
               valid_dataset=valid_set,
               device=torch.device("cuda"),
               config=model.get_conf(),
               save_directory=save_dir,
               epochs=100,
               batchsize=14,
               gradient_accumulation=5,
               epochs_per_save=10,
               spemb=False,
               lang="en",
               lr=0.005,
               warmup_steps=8000)

    ############################################################################
    save_dir = os.path.join("Models", "TransformerTTS", "SingleSpeaker", "lr01")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model = Transformer(idim=134, odim=80, spk_embed_dim=None)
    train_loop(net=model,
               train_dataset=train_set,
               valid_dataset=valid_set,
               device=torch.device("cuda"),
               config=model.get_conf(),
               save_directory=save_dir,
               epochs=100,
               batchsize=14,
               gradient_accumulation=5,
               epochs_per_save=10,
               spemb=False,
               lang="en",
               lr=0.01,
               warmup_steps=8000)

    ############################################################################
    save_dir = os.path.join("Models", "TransformerTTS", "SingleSpeaker", "decoder_concat")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model = Transformer(idim=134, odim=80, spk_embed_dim=None, decoder_concat_after=True)
    train_loop(net=model,
               train_dataset=train_set,
               valid_dataset=valid_set,
               device=torch.device("cuda"),
               config=model.get_conf(),
               save_directory=save_dir,
               epochs=100,
               batchsize=14,
               gradient_accumulation=5,
               epochs_per_save=10,
               spemb=False,
               lang="en",
               lr=0.001,
               warmup_steps=8000)

    ############################################################################
    save_dir = os.path.join("Models", "TransformerTTS", "SingleSpeaker", "red2")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model = Transformer(idim=134, odim=80, spk_embed_dim=None, reduction_factor=2)
    train_loop(net=model,
               train_dataset=train_set,
               valid_dataset=valid_set,
               device=torch.device("cuda"),
               config=model.get_conf(),
               save_directory=save_dir,
               epochs=100,
               batchsize=14,
               gradient_accumulation=5,
               epochs_per_save=10,
               spemb=False,
               lang="en",
               lr=0.001,
               warmup_steps=8000)
