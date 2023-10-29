"""
Example script for fine-tuning the pretrained model to your own data.

Comments in ALL CAPS are instructions
"""

import time

import torch
import wandb
from torch.utils.data import ConcatDataset

from Architectures.ToucanTTS.ToucanTTS import ToucanTTS
from Architectures.ToucanTTS.toucantts_train_loop_arbiter import train_loop
from Utility.corpus_preparation import prepare_tts_corpus
from Utility.path_to_transcript_dicts import *
from Utility.storage_config import MODELS_DIR
from Utility.storage_config import PREPROCESSING_DIR


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume, use_wandb, wandb_resume_id, gpu_count):
    if gpu_id == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    assert gpu_count == 1  # distributed finetuning is not supported

    # IF YOU'RE ADDING A NEW LANGUAGE, YOU MIGHT NEED TO ADD HANDLING FOR IT IN Preprocessing/TextFrontend.py

    print("Preparing")

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join(MODELS_DIR, "ToucanTTS_FinetuningExample")  # RENAME TO SOMETHING MEANINGFUL FOR YOUR DATA
    os.makedirs(save_dir, exist_ok=True)

    train_data = prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_integration_test(),
                                    corpus_dir=os.path.join(PREPROCESSING_DIR, "integration_test"),
                                    lang="en")  # CHANGE THE TRANSCRIPT DICT, THE NAME OF THE CACHE DIRECTORY AND THE LANGUAGE TO YOUR NEEDS

    model = ToucanTTS()

    if use_wandb:
        wandb.init(
            name=f"{__name__.split('.')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}" if wandb_resume_id is None else None,
            id=wandb_resume_id,  # this is None if not specified in the command line arguments.
            resume="must" if wandb_resume_id is not None else None)

    print("Training model")
    train_loop(net=model,
               datasets=[train_data],
               device=device,
               save_directory=save_dir,
               batch_size=12,  # YOU MIGHT GET OUT OF MEMORY ISSUES ON SMALL GPUs, IF SO, DECREASE THIS.
               eval_lang="en",  # THE LANGUAGE YOUR PROGRESS PLOTS WILL BE MADE IN
               warmup_steps=500,
               lr=1e-5,  # if you have enough data (over ~1000 datapoints) you can increase this up to 1e-4 and it will still be stable, but learn quicker.
               # DOWNLOAD THESE INITIALIZATION MODELS FROM THE RELEASE PAGE OF THE GITHUB OR RUN THE DOWNLOADER SCRIPT TO GET THEM AUTOMATICALLY
               path_to_checkpoint=os.path.join(MODELS_DIR, "ToucanTTS_Meta", "best.pt") if resume_checkpoint is None else resume_checkpoint,
               path_to_embed_model=os.path.join(MODELS_DIR, "Embedding", "embedding_function.pt"),
               fine_tune=True if resume_checkpoint is None and not resume else finetune,
               resume=resume,
               steps=5000,
               use_wandb=use_wandb,
               train_samplers=[torch.utils.data.RandomSampler(train_data)],
               gpu_count=1)
    if use_wandb:
        wandb.finish()
