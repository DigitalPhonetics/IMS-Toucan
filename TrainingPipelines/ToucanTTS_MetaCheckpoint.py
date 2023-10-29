import time

import torch
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

    base_dir = os.path.join(MODELS_DIR, "ToucanTTS_Meta")
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

    english_datasets = list()
    german_datasets = list()
    greek_datasets = list()
    spanish_datasets = list()
    finnish_datasets = list()
    russian_datasets = list()
    hungarian_datasets = list()
    dutch_datasets = list()
    french_datasets = list()
    portuguese_datasets = list()
    polish_datasets = list()
    italian_datasets = list()
    chinese_datasets = list()
    vietnamese_datasets = list()

    english_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_nancy,
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "Nancy"),
                                               lang="en",
                                               gpu_count=gpu_count,
                                               rank=rank))

    english_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_ljspeech,
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "LJSpeech"),
                                               lang="en",
                                               gpu_count=gpu_count,
                                               rank=rank))

    english_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_libritts_all_clean,
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "libri_all_clean"),
                                               lang="en",
                                               gpu_count=gpu_count,
                                               rank=rank))

    english_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_vctk,
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "vctk"),
                                               lang="en",
                                               gpu_count=gpu_count,
                                               rank=rank))

    english_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_CREMA_D,
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "crema_d"),
                                               lang="en",
                                               gpu_count=gpu_count,
                                               rank=rank))

    english_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_EmoV_DB,
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "emovdb"),
                                               lang="en",
                                               gpu_count=gpu_count,
                                               rank=rank))

    english_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_nvidia_hifitts,
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "hifi"),
                                               lang="en",
                                               gpu_count=gpu_count,
                                               rank=rank))

    english_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_RAVDESS,
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "ravdess"),
                                               lang="en",
                                               gpu_count=gpu_count,
                                               rank=rank))

    english_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_ESDS,
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "esds"),
                                               lang="en",
                                               gpu_count=gpu_count,
                                               rank=rank))

    german_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_karlsson,
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "Karlsson"),
                                              lang="de",
                                              gpu_count=gpu_count,
                                              rank=rank))

    german_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_eva,
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "Eva"),
                                              lang="de",
                                              gpu_count=gpu_count,
                                              rank=rank))

    german_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_hokus,
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "Hokus"),
                                              lang="de",
                                              gpu_count=gpu_count,
                                              rank=rank))

    german_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_bernd,
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "Bernd"),
                                              lang="de",
                                              gpu_count=gpu_count,
                                              rank=rank))

    german_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_hui_others,
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "hui_others"),
                                              lang="de",
                                              gpu_count=gpu_count,
                                              rank=rank))

    german_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_thorsten_2022_10(),
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "Thorsten"),
                                              lang="de",
                                              gpu_count=gpu_count,
                                              rank=rank))

    german_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_thorsten_emotional(),
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "Thorsten_emotional"),
                                              lang="de",
                                              gpu_count=gpu_count,
                                              rank=rank))

    german_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_thorsten_neutral(),
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "Thorsten_neutral"),
                                              lang="de",
                                              gpu_count=gpu_count,
                                              rank=rank))

    greek_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_css10el,
                                             corpus_dir=os.path.join(PREPROCESSING_DIR, "meta_Greek"),
                                             lang="el",
                                             gpu_count=gpu_count,
                                             rank=rank))

    spanish_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_spanish_blizzard_train,
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "spanish_blizzard"),
                                               lang="es",
                                               gpu_count=gpu_count,
                                               rank=rank))

    spanish_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_css10es,
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "meta_Spanish"),
                                               lang="es",
                                               gpu_count=gpu_count,
                                               rank=rank))

    spanish_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_mls_spanish,
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_spanish"),
                                               lang="es",
                                               gpu_count=gpu_count,
                                               rank=rank))

    finnish_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_css10fi,
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "meta_Finnish"),
                                               lang="fi",
                                               gpu_count=gpu_count,
                                               rank=rank))

    russian_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_css10ru,
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "meta_Russian"),
                                               lang="ru",
                                               gpu_count=gpu_count,
                                               rank=rank))

    hungarian_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_css10hu,
                                                 corpus_dir=os.path.join(PREPROCESSING_DIR, "meta_Hungarian"),
                                                 lang="hu",
                                                 gpu_count=gpu_count,
                                                 rank=rank))

    dutch_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_css10nl,
                                             corpus_dir=os.path.join(PREPROCESSING_DIR, "meta_Dutch"),
                                             lang="nl",
                                             gpu_count=gpu_count,
                                             rank=rank))

    dutch_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_mls_dutch,
                                             corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_dutch"),
                                             lang="nl",
                                             gpu_count=gpu_count,
                                             rank=rank))

    french_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_siwis_subset,
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "siwis"),
                                              lang="fr",
                                              gpu_count=gpu_count,
                                              rank=rank))

    french_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_blizzard2023_ad_silence_removed,
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "blizzard2023ad_silence_removed"),
                                              lang="fr",
                                              gpu_count=gpu_count,
                                              rank=rank))

    french_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_blizzard2023_neb_e_silence_removed,
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "blizzard2023neb_e_silence_removed"),
                                              lang="fr",
                                              gpu_count=gpu_count,
                                              rank=rank))

    french_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_blizzard2023_neb_silence_removed,
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "blizzard2023neb_silence_removed"),
                                              lang="fr",
                                              gpu_count=gpu_count,
                                              rank=rank))

    french_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_mls_french,
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_french"),
                                              lang="fr",
                                              gpu_count=gpu_count,
                                              rank=rank))  # this contains large portions of Canadian French

    portuguese_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_mls_portuguese,
                                                  corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_porto"),
                                                  lang="pt-br",
                                                  gpu_count=gpu_count,
                                                  rank=rank))

    polish_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_mls_polish,
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_polish"),
                                              lang="pl",
                                              gpu_count=gpu_count,
                                              rank=rank))

    italian_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_mls_italian,
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_italian"),
                                               lang="it",
                                               gpu_count=gpu_count,
                                               rank=rank))

    chinese_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_css10cmn,
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_chinese"),
                                               lang="cmn",
                                               gpu_count=gpu_count,
                                               rank=rank))

    chinese_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_aishell3,
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "aishell3"),
                                               lang="cmn",
                                               gpu_count=gpu_count,
                                               rank=rank))

    vietnamese_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_vietTTS,
                                                  corpus_dir=os.path.join(PREPROCESSING_DIR, "vietTTS"),
                                                  lang="vi",
                                                  gpu_count=gpu_count,
                                                  rank=rank))

    datasets.append(ConcatDataset(english_datasets))
    datasets.append(ConcatDataset(german_datasets))
    datasets.append(ConcatDataset(greek_datasets))
    datasets.append(ConcatDataset(spanish_datasets))
    datasets.append(ConcatDataset(finnish_datasets))
    datasets.append(ConcatDataset(russian_datasets))
    datasets.append(ConcatDataset(hungarian_datasets))
    datasets.append(ConcatDataset(dutch_datasets))
    datasets.append(ConcatDataset(french_datasets))
    datasets.append(ConcatDataset(portuguese_datasets))
    datasets.append(ConcatDataset(polish_datasets))
    datasets.append(ConcatDataset(italian_datasets))
    datasets.append(ConcatDataset(chinese_datasets))
    datasets.append(ConcatDataset(vietnamese_datasets))

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
               device=torch.device("cuda"),
               datasets=datasets,
               save_directory=meta_save_dir,
               path_to_checkpoint=resume_checkpoint,
               path_to_embed_model=os.path.join(MODELS_DIR, "Embedding", "embedding_function.pt"),
               resume=resume,
               fine_tune=finetune,
               steps=160000,
               use_wandb=use_wandb,
               train_samplers=train_samplers,
               gpu_count=gpu_count)
    if use_wandb:
        wandb.finish()
