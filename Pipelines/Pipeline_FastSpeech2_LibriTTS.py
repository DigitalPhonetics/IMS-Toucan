import os
import random

import torch

from FastSpeech2.FastSpeech2 import FastSpeech2
from FastSpeech2.FastSpeechDataset import FastSpeechDataset
from FastSpeech2.fastspeech2_train_loop import train_loop
from Utility.path_to_transcript_dicts import build_path_to_transcript_dict_libritts


def run(gpu_id, resume_checkpoint, finetune):
    if gpu_id == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")

    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
        device = torch.device("cuda")

    torch.manual_seed(13)
    random.seed(13)

    print("Preparing")
    cache_dir = os.path.join("Corpora", "LibriTTS")
    save_dir = os.path.join("Models", "FastSpeech2_LibriTTS")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path_to_transcript_dict = build_path_to_transcript_dict_libritts()

    train_set = FastSpeechDataset(path_to_transcript_dict,
                                  train=True,
                                  acoustic_model_name="TransformerTTS_LibriTTS/best.pt",
                                  cache_dir=cache_dir,
                                  lang="en",
                                  min_len_in_seconds=1,
                                  max_len_in_seconds=10,
                                  device=device,
                                  spemb=True,
                                  diagonal_attention_head_id=16)
    valid_set = FastSpeechDataset(path_to_transcript_dict,
                                  train=False,
                                  acoustic_model_name="TransformerTTS_LibriTTS/best.pt",
                                  cache_dir=cache_dir,
                                  lang="en",
                                  min_len_in_seconds=1,
                                  max_len_in_seconds=10,
                                  device=device,
                                  spemb=True,
                                  diagonal_attention_head_id=16)

    model = FastSpeech2(idim=133, odim=80, spk_embed_dim=256)

    print("Training model")
    train_loop(net=model,
               train_dataset=train_set,
               valid_dataset=valid_set,
               device=device,
               save_directory=save_dir,
               steps=400000,
               batch_size=32,
               gradient_accumulation=1,
               epochs_per_save=10,
               use_speaker_embedding=True,
               lang="en",
               lr=0.05,
               warmup_steps=8000,
               path_to_checkpoint=resume_checkpoint,
               fine_tune=finetune)
