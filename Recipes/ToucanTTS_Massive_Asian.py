import time

import torch
import wandb

from Utility.path_to_transcript_dicts import *


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume, use_wandb, wandb_resume_id, gpu_count):
    from torch.utils.data import ConcatDataset

    from Modules.ToucanTTS.ToucanTTS import ToucanTTS
    from Modules.ToucanTTS.toucantts_train_loop_arbiter import train_loop
    from Utility.corpus_preparation import prepare_tts_corpus
    from Utility.storage_config import MODEL_DIR
    from Utility.storage_config import PREPROCESSING_DIR

    if gpu_id == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    print("Preparing")

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join(MODEL_DIR, "ToucanTTS_Asian")
    os.makedirs(save_dir, exist_ok=True)

    if gpu_count > 1:
        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(backend="nccl")
    else:
        rank = 0

    lang_to_datasets = dict()

    lang_to_datasets["cmn"] = list()

    lang_to_datasets["cmn"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_css10cmn,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_chinese"),
                                                      lang="cmn",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["cmn"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_aishell3,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "aishell3"),
                                                      lang="cmn",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["vie"] = list()

    lang_to_datasets["vie"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_VIVOS_viet,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "VIVOS_viet"),
                                                      lang="vie",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_id = "cmn"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_fleurs_mandarin(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_mandarin"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))

    lang_id = "fil"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_fleurs_filipino(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_filipino"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))

    lang_id = "kor"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_fleurs_korean(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_korean"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))

    lang_id = "lao"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_fleurs_lao(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_lao"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))

    lang_id = "xng"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_fleurs_mongolian(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_mongolian"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))

    lang_id = "zsm"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_fleurs_malay(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_malay"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))

    lang_id = "vie"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_fleurs_vietnamese(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_vietnamese"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))

    lang_to_datasets["jpn"] = list()

    lang_to_datasets["jpn"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_captain_japanese,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "captain_japanese"),
                                                      lang="jpn",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["jpn"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_jvs,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "jvs"),
                                                      lang="jpn",
                                                      gpu_count=gpu_count,
                                                      rank=rank))
    datasets = list()
    for lang in lang_to_datasets:
        datasets.append(ConcatDataset(lang_to_datasets[lang]))

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
    print("Training model")
    train_loop(net=model,
               datasets=datasets,
               batch_size=12,
               steps_per_checkpoint=1000,
               device=device,
               save_directory=save_dir,
               eval_lang="vie",
               path_to_checkpoint=resume_checkpoint,
               fine_tune=finetune,
               resume=resume,
               use_wandb=use_wandb,
               train_samplers=train_samplers,
               gpu_count=gpu_count)
    if use_wandb:
        wandb.finish()
