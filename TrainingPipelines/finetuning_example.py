"""
Example script for fine-tuning the pretrained model to your own data.

Comments in ALL CAPS are instructions
"""

import time

import torch
import wandb
from torch.utils.data import ConcatDataset

from TTSTrainingInterfaces.ToucanTTS.ToucanTTS import ToucanTTS
from TTSTrainingInterfaces.ToucanTTS.toucantts_train_loop_arbiter import train_loop
from Utility.corpus_preparation import prepare_tts_corpus
from Utility.path_to_transcript_dicts import *
from Utility.storage_config import MODELS_DIR
from Utility.storage_config import PREPROCESSING_DIR


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume, use_wandb, wandb_resume_id, distributed):
    if gpu_id == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # IF YOU'RE ADDING A NEW LANGUAGE, YOU MIGHT NEED TO ADD HANDLING FOR IT IN Preprocessing/TextFrontend.py

    print("Preparing")

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join(MODELS_DIR, "ToucanTTS_German_and_English")  # RENAME TO SOMETHING MEANINGFUL FOR YOUR DATA
    os.makedirs(save_dir, exist_ok=True)

    all_train_sets = list()  # YOU CAN HAVE MULTIPLE LANGUAGES, OR JUST ONE. JUST MAKE ONE ConcatDataset PER LANGUAGE AND ADD IT TO THE LIST.
    train_samplers = list()

    # =======================
    # =    German Data      =
    # =======================
    german_datasets = list()
    german_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_karlsson(),
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "Karlsson"),
                                              lang="de"))  # CHANGE THE TRANSCRIPT DICT, THE NAME OF THE CACHE DIRECTORY AND THE LANGUAGE TO YOUR NEEDS

    german_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_eva(),
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "Eva"),
                                              lang="de"))  # YOU CAN SIMPLY ADD MODE CORPORA AND DO THE SAME, BUT YOU DON'T HAVE TO, ONE IS ENOUGH

    all_train_sets.append(ConcatDataset(german_datasets))

    # ========================
    # =    English Data      =
    # ========================
    english_datasets = list()
    english_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_nancy(),
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "Nancy"),
                                               lang="en"))

    english_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_ljspeech(),
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "LJSpeech"),
                                               lang="en"))

    all_train_sets.append(ConcatDataset(english_datasets))

    model = ToucanTTS()

    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        for train_set in all_train_sets:
            train_samplers.append(torch.utils.data.DistributedSampler(train_set, shuffle=True))
        model.to(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        torch.distributed.barrier()
        # torch.cuda.synchronize()
    else:
        for train_set in all_train_sets:
            train_samplers.append(torch.utils.data.RandomSampler(train_set))

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
               lr=1e-5,  # if you have enough data (over ~1000 datapoints) you can increase this up to 1e-4 and it will still be stable, but learn quicker.
               # DOWNLOAD THESE INITIALIZATION MODELS FROM THE RELEASE PAGE OF THE GITHUB OR RUN THE DOWNLOADER SCRIPT TO GET THEM AUTOMATICALLY
               path_to_checkpoint=os.path.join(MODELS_DIR, "ToucanTTS_Meta", "best.pt") if resume_checkpoint is None else resume_checkpoint,
               path_to_embed_model=os.path.join(MODELS_DIR, "Embedding", "embedding_function.pt"),
               fine_tune=True if resume_checkpoint is None and not resume else finetune,
               resume=resume,
               steps=5000,
               use_wandb=use_wandb,
               train_samplers=train_samplers)
    if use_wandb:
        wandb.finish()
