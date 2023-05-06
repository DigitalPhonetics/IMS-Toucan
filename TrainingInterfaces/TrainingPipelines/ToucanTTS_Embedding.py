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
        save_dir = os.path.join(MODELS_DIR, "ToucanTTS_EmbeddingAda")
    os.makedirs(save_dir, exist_ok=True)

    datasets = list()

    datasets.append(prepare_fastspeech_corpus(transcript_dict={},
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "libri_all_clean"),
                                              lang="en"))

    datasets.append(prepare_fastspeech_corpus(transcript_dict={},
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "ravdess"),
                                              lang="en",
                                              ctc_selection=False))

    datasets.append(prepare_fastspeech_corpus(transcript_dict={},
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "esds"),
                                              lang="en",
                                              ctc_selection=False))

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

    model = ToucanTTS(use_conditional_layernorm_embedding_integration=True)  # if we set this to true, a different embedding integration method will be used to give us a better embedding function
    if use_wandb:
        wandb.init(
            name=f"{__name__.split('.')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}" if wandb_resume_id is None else None,
            id=wandb_resume_id,  # this is None if not specified in the command line arguments.
            resume="must" if wandb_resume_id is not None else None)
    print("Training model")
    for _ in range(5):
        train_loop(net=model,
                   datasets=[train_set],
                   device=device,
                   save_directory=save_dir,
                   eval_lang="en",
                   path_to_checkpoint=resume_checkpoint,
                   path_to_embed_model=f"{save_dir}/embedding_function.pt",  # if we set this to None, we train the embedding function jointly from scratch
                   fine_tune=finetune,
                   steps=50000,
                   resume=resume,
                   postnet_start_steps=50000,  # don't need this, because the gradient is cut out.
                   use_wandb=use_wandb,
                   train_embed=True)  # we want to train the embedding function
    if use_wandb:
        wandb.finish()
