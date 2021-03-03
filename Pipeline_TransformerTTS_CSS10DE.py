"""
Train an autoregressive Transformer TTS model on the German single speaker dataset by Hokuspokus
"""

import os
import random
import warnings

import torch

from TransformerTTS.TransformerTTS import Transformer
from TransformerTTS.TransformerTTSDataset import TransformerTTSDataset
from TransformerTTS.transformer_tts_train_loop import train_loop

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
            path_to_transcript["Corpora/CSS10/" + line.split("|")[0]] = line.split("|")[2]
    return path_to_transcript


if __name__ == '__main__':
    print("Preparing")
    path_to_transcript_dict = build_path_to_transcript_dict()
    css10_train = TransformerTTSDataset(path_to_transcript_dict,
                                        train=True,
                                        load=True,
                                        cache_dir=os.path.join("Corpora", "CSS10"),
                                        lang="de",
                                        min_len=50000,
                                        max_len=230000)
    css10_valid = TransformerTTSDataset(path_to_transcript_dict,
                                        train=False,
                                        load=True,
                                        cache_dir=os.path.join("Corpora", "CSS10"),
                                        lang="de",
                                        min_len=50000,
                                        max_len=230000)
    model = Transformer(idim=132, odim=80, spk_embed_dim=None)
    if not os.path.exists("Models/TransformerTTS/SingleSpeaker/CSS10_DE"):
        os.makedirs("Models/TransformerTTS/SingleSpeaker/CSS10_DE")
    print("Training model")
    train_loop(net=model,
               train_dataset=css10_train,
               eval_dataset=css10_valid,
               device=torch.device("cuda:5"),
               config=model.get_conf(),
               save_directory="Models/TransformerTTS/SingleSpeaker/CSS10_DE",
               epochs=3000,  # just kill the process at some point
               batchsize=64,
               gradient_accumulation=1)
