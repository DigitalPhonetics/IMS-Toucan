import random

import torch

from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.fastspeech2_train_loop import train_loop
from Utility.corpus_preparation import prepare_fastspeech_corpus
from Utility.path_to_transcript_dicts import *


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume):
    if gpu_id == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")

    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
        device = torch.device("cuda")

    torch.manual_seed(131714)
    random.seed(131714)
    torch.random.manual_seed(131714)

    print("Preparing")

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join("Models", "FastSpeech2_LibriTTS_asr_out")
    os.makedirs(save_dir, exist_ok=True)

    train_set = prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_libritts_asr_out(),
                                          corpus_dir=os.path.join("Corpora", "libri_asr_out"),
                                          lang="en")

    model = FastSpeech2(lang_embs=None)

    print("Training model")
    train_loop(net=model,
               train_dataset=train_set,
               device=device,
               save_directory=save_dir,
               steps=500000,
               batch_size=32,
               lang="en",
               lr=0.0001,
               warmup_steps=4000,
               path_to_checkpoint="Models/FastSpeech2_LibriTTS/best.pt",
               fine_tune=True,
               resume=resume)
