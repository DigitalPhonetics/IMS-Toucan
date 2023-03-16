"""
Example script for fine-tuning the pretrained model to your own data.

Comments in ALL CAPS are instructions
"""

import time

import torch
import wandb
from torch.utils.data import ConcatDataset

from TrainingInterfaces.Text_to_Spectrogram.PortaSpeech.PortaSpeech import PortaSpeech
from TrainingInterfaces.Text_to_Spectrogram.PortaSpeech.portaspeech_train_loop_arbiter import train_loop
from Utility.corpus_preparation import prepare_fastspeech_corpus
from Utility.path_to_transcript_dicts import *
from Utility.storage_config import MODELS_DIR
from Utility.storage_config import PREPROCESSING_DIR


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

    # IF YOU'RE ADDING A NEW LANGUAGE, YOU MIGHT NEED TO ADD HANDLING FOR IT IN Preprocessing/TextFrontend.py

    print("Preparing")

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join(MODELS_DIR, "PortaSpeech_German_and_English")  # RENAME TO SOMETHING MEANINGFUL FOR YOUR DATA
    os.makedirs(save_dir, exist_ok=True)

    all_train_sets = list()  # YOU CAN HAVE MULTIPLE LANGUAGES, OR JUST ONE. JUST MAKE ONE ConcatDataset PER LANGUAGE AND ADD IT TO THE LIST.

    # =======================
    # =    German Data      =
    # =======================
    german_datasets = list()
    german_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_karlsson(),
                                                     corpus_dir=os.path.join(PREPROCESSING_DIR, "Karlsson"),
                                                     lang="de"))  # CHANGE THE TRANSCRIPT DICT, THE NAME OF THE CACHE DIRECTORY AND THE LANGUAGE TO YOUR NEEDS

    german_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_eva(),
                                                     corpus_dir=os.path.join(PREPROCESSING_DIR, "Eva"),
                                                     lang="de"))  # YOU CAN SIMPLY ADD MODE CORPORA AND DO THE SAME, BUT YOU DON'T HAVE TO, ONE IS ENOUGH

    all_train_sets.append(ConcatDataset(german_datasets))

    # ========================
    # =    English Data      =
    # ========================
    english_datasets = list()
    english_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_nancy(),
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "Nancy"),
                                                      lang="en"))

    english_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_ljspeech(),
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "LJSpeech"),
                                                      lang="en"))

    all_train_sets.append(ConcatDataset(english_datasets))

    model = PortaSpeech()
    if use_wandb:
        wandb.init(
            name=f"{__name__.split('.')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}" if wandb_resume_id is None else None,
            id=wandb_resume_id, resume="must" if wandb_resume_id is not None else None)
    print("Training model")
    train_loop(net=model,
               datasets=all_train_sets,
               device=device,
               save_directory=save_dir,
               batch_size=12,  # YOU MIGHT GET OUT OF MEMORY ISSUES ON SMALL GPUs, IF SO, DECREASE THIS.
               eval_lang="de",  # THE LANGUAGE YOUR PROGRESS PLOTS WILL BE MADE IN
               warmup_steps=500,
               # DOWNLOAD THESE INITIALIZATION MODELS FROM THE RELEASE PAGE OF THE GITHUB OR RUN THE DOWNLOADER SCRIPT TO GET THEM AUTOMATICALLY
               path_to_checkpoint=os.path.join(MODELS_DIR, "PortaSpeech_Meta", "best.pt") if resume_checkpoint is None else resume_checkpoint,
               path_to_embed_model=os.path.join(MODELS_DIR, "Embedding", "embedding_function.pt"),
               fine_tune=True if resume_checkpoint is None else finetune,
               resume=resume,
               phase_1_steps=5000,
               phase_2_steps=1000,
               use_wandb=use_wandb)
    if use_wandb:
        wandb.finish()
