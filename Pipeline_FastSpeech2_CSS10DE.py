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

warnings.filterwarnings("ignore")

torch.manual_seed(17)
random.seed(17)


def build_path_to_transcript_dict():
    path_to_transcript = dict()
    with open("Corpora/CSS10/transcript.txt", encoding="utf8") as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            path_to_transcript[line.split("|")[0]] = line.split("|")[2]
    return path_to_transcript


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def show_model(model):
    print(model)
    print("\n\nNumber of Parameters: {}".format(count_parameters(model)))


if __name__ == '__main__':
    print("Preparing")
    device = torch.device("cuda")
    path_to_transcript_dict = build_path_to_transcript_dict()
    css10_train = FastSpeechDataset(path_to_transcript_dict, train=True,
                                    acoustic_model_name="Transformer_German_Single.pt")
    css10_valid = FastSpeechDataset(path_to_transcript_dict, train=False,
                                    acoustic_model_name="Transformer_German_Single.pt")
    model = FastSpeech2(idim=132, odim=80, spk_embed_dim=None).to(device)
    if not os.path.exists("Models/FastSpeech2/SingleSpeaker/CSS10"):
        os.makedirs("Models/FastSpeech2/SingleSpeaker/CSS10")
    print("Training model")
    train_loop(net=model,
               train_dataset=css10_train,
               eval_dataset=css10_valid,
               device=device,
               config=model.get_conf(),
               save_directory="Models/FastSpeech2/SingleSpeaker/CSS10",
               epochs=3000,  # just kill the process at some point
               batchsize=16,
               gradient_accumulation=4)
