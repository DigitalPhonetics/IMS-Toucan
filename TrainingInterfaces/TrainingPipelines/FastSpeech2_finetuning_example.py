import random

import torch
from torch.utils.data import ConcatDataset

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
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
        device = torch.device("cuda")

    torch.manual_seed(131714)
    random.seed(131714)
    torch.random.manual_seed(131714)

    print("Preparing")

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join("Models", "FastSpeech2_German")  # KEEP THE 'FastSpeech2_' PART BUT CHANGE THE MODEL ID TO SOMETHING MEANINGFUL FOR YOUR DATA
    os.makedirs(save_dir, exist_ok=True)

    datasets = list()
    datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_karlsson(),
                                              corpus_dir=os.path.join("Corpora", "Karlsson"),
                                              lang="de"))  # CHANGE THE TRANSCRIPT DICT, THE NAME OF THE CACHE DIRECTORY AND THE LANGUAGE TO YOUR NEEDS

    datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_eva(),
                                              corpus_dir=os.path.join("Corpora", "Eva"),
                                              lang="de"))  # YOU CAN SIMPLY ADD MODE CORPORA AND DO THE SAME, BUT YOU DON'T HAVE TO, ONE IS ENOUGH

    train_set = ConcatDataset(datasets)

    print("Training model")
    train_loop(net=FastSpeech2(),
               train_dataset=train_set,
               device=device,
               save_directory=save_dir,
               batch_size=12,  # YOU MIGHT GET OUT OF MEMORY ISSUES ON SMALL GPUs, IF SO, DECREASE THIS
               lang="de",  # CHANGE THIS TO THE LANGUAGE YOU'RE TRAINING ON
               lr=0.001,
               epochs_per_save=1,
               warmup_steps=4000,
               # DOWNLOAD THIS INITIALIZATION MODEL FROM THE RELEASE PAGE OF THE GITHUB REPO
               path_to_checkpoint="Models/FastSpeech2_Meta/best.pt" if resume_checkpoint is None else resume_checkpoint,
               fine_tune=True if resume_checkpoint is None else finetune,
               resume=resume)
