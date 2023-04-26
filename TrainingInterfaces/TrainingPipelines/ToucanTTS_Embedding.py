import time

import torch
import wandb
from torch.utils.data import ConcatDataset

from TrainingInterfaces.Text_to_Spectrogram.ToucanTTS.ToucanTTS import ToucanTTS
from TrainingInterfaces.Text_to_Spectrogram.ToucanTTS.toucantts_train_loop_arbiter import train_loop
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
        save_dir = os.path.join(MODELS_DIR, "ToucanTTS_Embedding")
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

    datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_vctk(),
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "vctk"),
                                              lang="en"))

    datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_CREMA_D(),
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "crema_d"),
                                              lang="en"))

    datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_EmoV_DB(),
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "emovdb"),
                                              lang="en"))

    train_set = ConcatDataset(datasets)

    model = ToucanTTS()
    if use_wandb:
        wandb.init(
            name=f"{__name__.split('.')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}" if wandb_resume_id is None else None,
            id=wandb_resume_id,  # this is None if not specified in the command line arguments.
            resume="must" if wandb_resume_id is not None else None)
    print("Training model")
    train_loop(net=model,
               datasets=[train_set],
               device=device,
               save_directory=save_dir,
               eval_lang="en",
               path_to_checkpoint=resume_checkpoint,
               path_to_embed_model=None,  # if we set this to None, we train the embedding function jointly
               fine_tune=finetune,
               steps=160000,
               resume=resume,
               use_wandb=use_wandb)
    if use_wandb:
        wandb.finish()
