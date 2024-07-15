import time

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

    print("Preparing")

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join(MODELS_DIR, "ToucanTTS_English_v2")
    os.makedirs(save_dir, exist_ok=True)

    if gpu_count > 1:
        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(backend="nccl")
    else:
        rank = 0

    datasets = list()

    datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_nancy,
                                       corpus_dir=os.path.join(PREPROCESSING_DIR, "Nancy"),
                                       lang="eng",
                                       gpu_count=gpu_count,
                                       rank=rank))
    chunk_count = 100
    chunks = split_dictionary_into_chunks(build_path_to_transcript_dict_mls_english(), split_n=chunk_count)
    for index in range(chunk_count):
        datasets.append(prepare_tts_corpus(transcript_dict=chunks[index],
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, f"mls_english_chunk_{index}"),
                                           lang="eng",
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_ryanspeech,
                                       corpus_dir=os.path.join(PREPROCESSING_DIR, "Ryan"),
                                       lang="eng",
                                       gpu_count=gpu_count,
                                       rank=rank))

    datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_ljspeech,
                                       corpus_dir=os.path.join(PREPROCESSING_DIR, "LJSpeech"),
                                       lang="eng",
                                       gpu_count=gpu_count,
                                       rank=rank))

    datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_libritts_all_clean,
                                       corpus_dir=os.path.join(PREPROCESSING_DIR, "libri_all_clean"),
                                       lang="eng",
                                       gpu_count=gpu_count,
                                       rank=rank))

    datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_vctk,
                                       corpus_dir=os.path.join(PREPROCESSING_DIR, "vctk"),
                                       lang="eng",
                                       gpu_count=gpu_count,
                                       rank=rank))

    datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_nvidia_hifitts,
                                       corpus_dir=os.path.join(PREPROCESSING_DIR, "hifi"),
                                       lang="eng",
                                       gpu_count=gpu_count,
                                       rank=rank))

    datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_CREMA_D,
                                       corpus_dir=os.path.join(PREPROCESSING_DIR, "cremad"),
                                       lang="eng",
                                       gpu_count=gpu_count,
                                       rank=rank))

    datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_EmoV_DB,
                                       corpus_dir=os.path.join(PREPROCESSING_DIR, "emovdb"),
                                       lang="eng",
                                       gpu_count=gpu_count,
                                       rank=rank))

    datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_RAVDESS,
                                       corpus_dir=os.path.join(PREPROCESSING_DIR, "ravdess"),
                                       lang="eng",
                                       gpu_count=gpu_count,
                                       rank=rank))

    datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_ESDS,
                                       corpus_dir=os.path.join(PREPROCESSING_DIR, "esds"),
                                       lang="eng",
                                       gpu_count=gpu_count,
                                       rank=rank))

    datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_blizzard_2013,
                                       corpus_dir=os.path.join(PREPROCESSING_DIR, "blizzard2013"),
                                       lang="eng",
                                       gpu_count=gpu_count,
                                       rank=rank))

    datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_jenny,
                                       corpus_dir=os.path.join(PREPROCESSING_DIR, "jenny"),
                                       lang="eng",
                                       gpu_count=gpu_count,
                                       rank=rank))

    datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_ears,
                                       corpus_dir=os.path.join(PREPROCESSING_DIR, "ears"),
                                       lang="eng",
                                       gpu_count=gpu_count,
                                       rank=rank))

    train_set = ConcatDataset(datasets)

    model = ToucanTTS()

    if gpu_count > 1:
        model.to(rank)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True,
        )
        torch.distributed.barrier()
    train_sampler = torch.utils.data.RandomSampler(train_set)

    if use_wandb:
        if rank == 0:
            wandb.init(
                name=f"{__name__.split('.')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}" if wandb_resume_id is None else None,
                id=wandb_resume_id,  # this is None if not specified in the command line arguments.
                resume="must" if wandb_resume_id is not None else None)
    print("Training model")
    train_loop(net=model,
               datasets=[train_set],
               batch_size=16,
               steps_per_checkpoint=1000,
               device=device,
               save_directory=save_dir,
               eval_lang="eng",
               path_to_checkpoint=resume_checkpoint,
               fine_tune=finetune,
               resume=resume,
               use_wandb=use_wandb,
               train_samplers=[train_sampler],
               gpu_count=gpu_count)
    if use_wandb:
        wandb.finish()
