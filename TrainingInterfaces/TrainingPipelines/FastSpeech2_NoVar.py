import time

import torch
import torch.multiprocessing
import wandb

from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.fastspeech2_train_loop import train_loop
from Utility.corpus_preparation import prepare_fastspeech_corpus
from Utility.path_to_transcript_dicts import *


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume, use_wandb, wandb_resume_id):
    # It is not recommended training this yourself or to finetune this, but you can.
    # The recommended use is to download the pretrained model from the github release
    # page and finetune to your desired data similar to how it is showcased in
    # FastSpeech2_finetuning_example.py

    torch.manual_seed(131714)
    random.seed(131714)
    torch.random.manual_seed(131714)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"

    datasets = list()

    base_dir = os.path.join("Models", "FastSpeech2_NonVariational")
    if model_dir is not None:
        meta_save_dir = model_dir
    else:
        meta_save_dir = base_dir
    os.makedirs(meta_save_dir, exist_ok=True)

    print("Preparing")

    train_set = prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_libritts_all_clean(),
                                          corpus_dir=os.path.join("Corpora", "libri_all_clean"),
                                          lang="en")
    model = FastSpeech2(variational=False)
    if use_wandb:
        wandb.init(
            name=f"{__name__.split('.')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}" if wandb_resume_id is None else None,
            id=wandb_resume_id,  # this is None if not specified in the command line arguments.
            resume="must" if wandb_resume_id is not None else None)
    train_loop(net=model,
               device=torch.device("cuda"),
               train_dataset=train_set,
               batch_size=32,
               save_directory=meta_save_dir,
               phase_1_steps=50000,
               phase_2_steps=50000,
               lr=0.001,
               path_to_checkpoint=resume_checkpoint,
               path_to_embed_model="Models/Embedding/embedding_function.pt",
               resume=resume,
               use_wandb=use_wandb)
    if use_wandb:
        wandb.finish()
