"""

STAGE 1: Train a model that can handle multiple speakers in one language first.

"""

import time

import torch.multiprocessing
import wandb
from torch.utils.data import ConcatDataset

from Architectures.ToucanTTS.ToucanTTS import ToucanTTS
from Architectures.ToucanTTS.toucantts_train_loop_arbiter import train_loop
from Utility.corpus_preparation import prepare_tts_corpus
from Utility.path_to_transcript_dicts import *
from Utility.storage_config import MODELS_DIR
from Utility.storage_config import PREPROCESSING_DIR


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume, use_wandb, wandb_resume_id, gpu_count):
    # It is not recommended training this yourself or to finetune this, but you can.
    # The recommended use is to download the pretrained model from the GitHub release
    # page and finetune to your desired data

    datasets = list()

    base_dir = os.path.join(MODELS_DIR, "ToucanTTS_MassiveDataBigModel_stage1")
    if model_dir is not None:
        meta_save_dir = model_dir
    else:
        meta_save_dir = base_dir
    os.makedirs(meta_save_dir, exist_ok=True)

    print("Preparing")

    if gpu_count > 1:
        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(backend="nccl")
    else:
        rank = 0

    lang_to_datasets = dict()

    # ENGLISH

    lang_to_datasets["eng"] = list()

    lang_to_datasets["eng"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_nancy,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "Nancy"),
                                                      lang="eng",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    chunk_count = 50
    chunks = split_dictionary_into_chunks(build_path_to_transcript_dict_mls_english(), split_n=chunk_count)
    for index in range(chunk_count):
        lang_to_datasets["eng"].append(prepare_tts_corpus(transcript_dict=chunks[index],
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, f"mls_english_chunk_{index}"),
                                                          lang="eng",
                                                          gpu_count=gpu_count,
                                                          rank=rank))

    lang_to_datasets["eng"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_ryanspeech,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "Ryan"),
                                                      lang="eng",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["eng"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_ljspeech,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "LJSpeech"),
                                                      lang="eng",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["eng"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_libritts_all_clean,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "libri_all_clean"),
                                                      lang="eng",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["eng"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_RAVDESS,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "ravdess"),
                                                      lang="eng",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["eng"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_blizzard_2013,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "blizzard2013"),
                                                      lang="eng",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["eng"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_jenny,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "jenny"),
                                                      lang="eng",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    for lang in lang_to_datasets:
        datasets.append(ConcatDataset(lang_to_datasets[lang]))
    re_ordered_datasets = list()

    print(f"\n\nTraining jointly on {len(datasets)} languages in a setup of {len(re_ordered_datasets)} tasks! Good luck!\n\n")
    print(lang_to_datasets.keys())
    print("\n\n")

    model = ToucanTTS()

    train_samplers = list()
    if gpu_count > 1:
        model.to(rank)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True,
        )
        torch.distributed.barrier()
    for train_set in datasets:
        train_samplers.append(torch.utils.data.RandomSampler(train_set))

    if use_wandb:
        if rank == 0:
            wandb.init(
                name=f"{__name__.split('.')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}" if wandb_resume_id is None else None,
                id=wandb_resume_id,  # this is None if not specified in the command line arguments.
                resume="must" if wandb_resume_id is not None else None)
    train_loop(net=model,
               batch_size=8,
               warmup_steps=4000,
               device=torch.device("cuda"),
               datasets=datasets[0],
               save_directory=meta_save_dir,
               path_to_checkpoint=resume_checkpoint,
               path_to_embed_model=os.path.join(MODELS_DIR, "Embedding", "embedding_function.pt"),
               resume=resume,
               fine_tune=finetune,
               steps=30000,
               steps_per_checkpoint=1000,
               lr=0.0001,
               use_wandb=use_wandb,
               train_samplers=train_samplers,
               gpu_count=gpu_count)
    if use_wandb:
        wandb.finish()
