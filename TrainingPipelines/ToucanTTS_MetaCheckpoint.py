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

    chunk_count = 50
    chunks = split_dictionary_into_chunks(build_path_to_transcript_dict_mls_english(), split_n=chunk_count)
    for index in range(chunk_count):
        english_datasets.append(prepare_tts_corpus(transcript_dict=chunks[index],
                                                   corpus_dir=os.path.join(PREPROCESSING_DIR, f"mls_english_chunk_{index}"),
                                                   lang="eng",
                                                   gpu_count=gpu_count,
                                                   rank=rank))

    english_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_nancy,
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "Nancy"),
                                               lang="eng",
                                               gpu_count=gpu_count,
                                               rank=rank))

    english_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_ryanspeech,
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "Ryan"),
                                               lang="eng",
                                               gpu_count=gpu_count,
                                               rank=rank))

    english_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_ljspeech,
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "LJSpeech"),
                                               lang="eng",
                                               gpu_count=gpu_count,
                                               rank=rank))

    english_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_libritts_all_clean,
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "libri_all_clean"),
                                               lang="eng",
                                               gpu_count=gpu_count,
                                               rank=rank))

    english_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_vctk,
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "vctk"),
                                               lang="eng",
                                               gpu_count=gpu_count,
                                               rank=rank))

    english_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_nvidia_hifitts,
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "hifi"),
                                               lang="eng",
                                               gpu_count=gpu_count,
                                               rank=rank))

    english_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_CREMA_D,
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "cremad"),
                                               lang="eng",
                                               gpu_count=gpu_count,
                                               rank=rank))

    english_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_EmoV_DB,
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "emovdb"),
                                               lang="eng",
                                               gpu_count=gpu_count,
                                               rank=rank))

    english_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_RAVDESS,
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "ravdess"),
                                               lang="eng",
                                               gpu_count=gpu_count,
                                               rank=rank))

    english_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_ESDS,
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "esds"),
                                               lang="eng",
                                               gpu_count=gpu_count,
                                               rank=rank))

    english_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_blizzard_2013,
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "blizzard2013"),
                                               lang="eng",
                                               gpu_count=gpu_count,
                                               rank=rank))

    english_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_jenny,
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "jenny"),
                                               lang="eng",
                                               gpu_count=gpu_count,
                                               rank=rank))

    # GERMAN

    german_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_karlsson,
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "Karlsson"),
                                              lang="deu",
                                              gpu_count=gpu_count,
                                              rank=rank))

    german_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_eva,
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "Eva"),
                                              lang="deu",
                                              gpu_count=gpu_count,
                                              rank=rank))

    german_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_hokus,
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "Hokus"),
                                              lang="deu",
                                              gpu_count=gpu_count,
                                              rank=rank))

    german_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_bernd,
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "Bernd"),
                                              lang="deu",
                                              gpu_count=gpu_count,
                                              rank=rank))

    german_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_friedrich,
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "Friedrich"),
                                              lang="deu",
                                              gpu_count=gpu_count,
                                              rank=rank))

    german_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_hui_others,
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "hui_others"),
                                              lang="deu",
                                              gpu_count=gpu_count,
                                              rank=rank))

    german_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_thorsten_emotional(),
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "thorsten_emotional"),
                                              lang="deu",
                                              gpu_count=gpu_count,
                                              rank=rank))

    german_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_thorsten_neutral(),
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "thorsten_neutral"),
                                              lang="deu",
                                              gpu_count=gpu_count,
                                              rank=rank))

    german_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_thorsten_2022_10(),
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "thorsten_2022"),
                                              lang="deu",
                                              gpu_count=gpu_count,
                                              rank=rank))

    chunk_count = 10
    chunks = split_dictionary_into_chunks(build_path_to_transcript_dict_mls_german(), split_n=chunk_count)
    for index in range(chunk_count):
        german_datasets.append(prepare_tts_corpus(transcript_dict=chunks[index],
                                                  corpus_dir=os.path.join(PREPROCESSING_DIR, f"mls_german_chunk_{index}"),
                                                  lang="deu",
                                                  gpu_count=gpu_count,
                                                  rank=rank))

    # FRENCH

    french_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_css10fr,
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_French"),
                                              lang="fra",
                                              gpu_count=gpu_count,
                                              rank=rank))

    french_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_mls_french,
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_french"),
                                              lang="fra",
                                              gpu_count=gpu_count,
                                              rank=rank))

    french_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_blizzard2023_ad_silence_removed,
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "ad_e"),
                                              lang="fra",
                                              gpu_count=gpu_count,
                                              rank=rank))

    french_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_blizzard2023_neb_silence_removed,
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "neb"),
                                              lang="fra",
                                              gpu_count=gpu_count,
                                              rank=rank))

    french_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_blizzard2023_neb_e_silence_removed,
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "neb_e"),
                                              lang="fra",
                                              gpu_count=gpu_count,
                                              rank=rank))

    french_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_synpaflex_norm_subset,
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "synpaflex"),
                                              lang="fra",
                                              gpu_count=gpu_count,
                                              rank=rank))

    french_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_siwis_subset,
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "siwis"),
                                              lang="fra",
                                              gpu_count=gpu_count,
                                              rank=rank))

    # SPANISH

    spanish_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_mls_spanish,
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_spanish"),
                                               lang="spa",
                                               gpu_count=gpu_count,
                                               rank=rank))

    spanish_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_css10es,
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Spanish"),
                                               lang="spa",
                                               gpu_count=gpu_count,
                                               rank=rank))

    spanish_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_spanish_blizzard_train,
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "spanish_blizzard"),
                                               lang="spa",
                                               gpu_count=gpu_count,
                                               rank=rank))

    # CHINESE

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

    # PORTUGUESE

    portuguese_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_mls_portuguese,
                                                  corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_porto"),
                                                  lang="por",
                                                  gpu_count=gpu_count,
                                                  rank=rank))

    # POLISH

    polish_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_mls_polish,
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_polish"),
                                              lang="pol",
                                              gpu_count=gpu_count,
                                              rank=rank))

    # ITALIAN

    italian_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_mls_italian,
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_italian"),
                                               lang="ita",
                                               gpu_count=gpu_count,
                                               rank=rank))

    # DUTCH

    dutch_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_mls_dutch,
                                             corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_dutch"),
                                             lang="nld",
                                             gpu_count=gpu_count,
                                             rank=rank))

    dutch_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_css10nl,
                                             corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Dutch"),
                                             lang="nld",
                                             gpu_count=gpu_count,
                                             rank=rank))

    # GREEK

    greek_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_css10el,
                                             corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Greek"),
                                             lang="ell",
                                             gpu_count=gpu_count,
                                             rank=rank))

    # FINNISH

    finnish_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_css10fi,
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Finnish"),
                                               lang="fin",
                                               gpu_count=gpu_count,
                                               rank=rank))

    # VIETNAMESE

    vietnamese_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_VIVOS_viet,
                                                  corpus_dir=os.path.join(PREPROCESSING_DIR, "VIVOS_viet"),
                                                  lang="vie",
                                                  gpu_count=gpu_count,
                                                  rank=rank))

    # RUSSIAN

    russian_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_css10ru,
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Russian"),
                                               lang="rus",
                                               gpu_count=gpu_count,
                                               rank=rank))

    # HUNGARIAN

    hungarian_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_css10hu,
                                                 corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Hungarian"),
                                                 lang="hun",
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
               resume=resume,
               fine_tune=finetune,
               steps=200000,
               steps_per_checkpoint=2000,
               lr=0.0001,
               use_wandb=use_wandb,
               train_samplers=train_samplers,
               gpu_count=gpu_count)
    if use_wandb:
        wandb.finish()