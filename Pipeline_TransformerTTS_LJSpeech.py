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

warnings.filterwarnings("ignore")

torch.manual_seed(13)
random.seed(13)


def build_path_to_transcript_dict():
    path_to_transcript = dict()
    for transcript_file in os.listdir("/mount/resources/speech/corpora/LJSpeech/16kHz/txt"):
        with open("/mount/resources/speech/corpora/LJSpeech/16kHz/txt/" + transcript_file, 'r', encoding='utf8') as tf:
            transcript = tf.read()
        wav_path = "/mount/resources/speech/corpora/LJSpeech/16kHz/wav/" + transcript_file.split(".")[0] + ".wav"
        path_to_transcript[wav_path] = transcript
    return path_to_transcript


if __name__ == '__main__':
    print("Preparing")
    cache_dir = os.path.join("Corpora", "LJSpeech")
    save_dir = os.path.join("Models", "TransformerTTS", "SingleSpeaker", "LJSpeech", "espnet_settings")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path_to_transcript_dict = build_path_to_transcript_dict()

    train_set = TransformerTTSDataset(path_to_transcript_dict,
                                      train=True,
                                      cache_dir=cache_dir,
                                      lang="en",
                                      min_len=0,
                                      max_len=170000)
    valid_set = TransformerTTSDataset(path_to_transcript_dict,
                                      train=False,
                                      cache_dir=cache_dir,
                                      lang="en",
                                      min_len=0,
                                      max_len=170000)

    model = Transformer(idim=131, odim=80, spk_embed_dim=None)

    print("Training model")
    train_loop(net=model,
               train_dataset=train_set,
               valid_dataset=valid_set,
               device=torch.device("cuda:1"),
               config=model.get_conf(),
               save_directory=save_dir,
               epochs=300000,  # just kill the process at some point
               batchsize=64,
               gradient_accumulation=1)
