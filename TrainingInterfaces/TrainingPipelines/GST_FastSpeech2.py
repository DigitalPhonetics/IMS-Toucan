"""
This is the setup with which the embedding model is trained. After the embedding model has been trained, it is only used in a frozen state.
"""

import time

import torch
import wandb
from torch.utils.data import ConcatDataset

from TrainingInterfaces.Spectrogram_to_Embedding.embedding_function_train_loop import train_loop
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2
from Utility.corpus_preparation import prepare_fastspeech_corpus
from Utility.path_to_transcript_dicts import *
from Utility.storage_config import MODELS_DIR
from Utility.storage_config import PREPROCESSING_DIR


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume, use_wandb, wandb_resume_id):
    if gpu_id == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    print("Preparing")

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join(MODELS_DIR, "FastSpeech2_Embedding")
    os.makedirs(save_dir, exist_ok=True)

    datasets = list()

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

    # for the next iteration, we should add an augmented noisy version of e.g. Nancy,
    # so the embedding learns to factor out noise

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
               use_wandb=use_wandb)
    if use_wandb:
        wandb.finish()
