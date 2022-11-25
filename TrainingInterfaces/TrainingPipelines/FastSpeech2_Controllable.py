import os
import time

import torch
import wandb
from torch.utils.data import ConcatDataset

from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.fastspeech2_train_loop import train_loop
from Utility.corpus_preparation import prepare_fastspeech_corpus
from Utility.path_to_transcript_dicts import *
from Utility.storage_config import MODELS_DIR, PREPROCESSING_DIR


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume, use_wandb, wandb_resume_id):
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
        save_dir = os.path.join(MODELS_DIR, "FastSpeech2_Controllable")
    os.makedirs(save_dir, exist_ok=True)

    datasets = list()
    datasets.append(prepare_fastspeech_corpus(transcript_dict={},
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "Nancy"),
                                              lang="en"))

    datasets.append(prepare_fastspeech_corpus(transcript_dict={},
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "ravdess"),
                                              lang="en",
                                              ctc_selection=False))

    datasets.append(prepare_fastspeech_corpus(transcript_dict={},
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "esds"),
                                              lang="en",
                                              ctc_selection=False))

    datasets.append(prepare_fastspeech_corpus(transcript_dict={},
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "libri_all_clean"),
                                              lang="en"))

    datasets.append(prepare_fastspeech_corpus(transcript_dict={},
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "vctk"),
                                              lang="en"))

    datasets.append(prepare_fastspeech_corpus(transcript_dict={},
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "LJSpeech"),
                                              lang="en"))
    train_set = ConcatDataset(datasets)

    model = FastSpeech2(lang_embs=None)
    if use_wandb:
        wandb.init(
            name=f"{__name__.split('.')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}" if wandb_resume_id is None else None,
            id=wandb_resume_id,  # this is None if not specified in the command line arguments.
            resume="must" if wandb_resume_id is not None else None)
    print("Training model")
    train_loop(net=model,
               train_dataset=train_set,
               device=device,
               save_directory=save_dir,
               batch_size=32,
               lang="en",
               lr=0.001,
               epochs_per_save=1,
               warmup_steps=4000,
               path_to_checkpoint=resume_checkpoint,
               fine_tune=finetune,
               resume=resume,
               phase_1_steps=150000,
               phase_2_steps=50000,
               use_wandb=use_wandb)
    if use_wandb:
        wandb.finish()
