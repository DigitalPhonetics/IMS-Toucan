import torch
from torch.utils.data import ConcatDataset

from TrainingInterfaces.Text_to_Spectrogram.AutoAligner.autoaligner_train_loop import train_loop as train_aligner
from Utility.corpus_preparation import prepare_aligner_corpus
from Utility.path_to_transcript_dicts import *
from Utility.storage_config import MODELS_DIR
from Utility.storage_config import PREPROCESSING_DIR


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume, use_wandb, wandb_resume_id):
    if gpu_id == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    print("Preparing")

    datasets = list()

    # ENGLISH

    chunk_count = 50
    chunks = split_dictionary_into_chunks(build_path_to_transcript_dict_mls_english(), split_n=chunk_count)
    for index in range(chunk_count):
        datasets.append(prepare_aligner_corpus(transcript_dict=chunks[index],
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, f"mls_english_chunk_{index}"),
                                               lang="en",
                                               device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_nancy,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "Nancy"),
                                           lang="en",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_elizabeth,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "Elizabeth"),
                                           lang="en",  # technically, she's british english, not american english
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_ryanspeech,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "Ryan"),
                                           lang="en",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_ljspeech,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "LJSpeech"),
                                           lang="en",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_libritts_all_clean,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "libri"),
                                           lang="en",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_vctk,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "vctk"),
                                           lang="en",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_nvidia_hifitts,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "hifi"),
                                           lang="en",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_CREMA_D,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "cremad"),
                                           lang="en",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_EmoV_DB,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "emovdb"),
                                           lang="en",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_RAVDESS,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "ravdess"),
                                           lang="en",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_ESDS,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "esds"),
                                           lang="en",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_blizzard_2013,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "blizzard2013"),
                                           lang="en",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_jenny,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "jenny"),
                                           lang="en",
                                           device=device))

    chunk_count = 30
    chunks = split_dictionary_into_chunks(build_path_to_transcript_dict_gigaspeech(), split_n=chunk_count)
    for index in range(chunk_count):
        datasets.append(prepare_aligner_corpus(transcript_dict=chunks[index],
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, f"gigaspeech_chunk_{index}"),
                                               lang="en",
                                               device=device))

    # GERMAN

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_karlsson,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "Karlsson"),
                                           lang="de",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_eva,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "Eva"),
                                           lang="de",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_hokus,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "Hokus"),
                                           lang="de",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_bernd,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "Bernd"),
                                           lang="de",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_friedrich,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "Friedrich"),
                                           lang="de",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_hui_others,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "hui_others"),
                                           lang="de",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_thorsten,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "Thorsten"),
                                           lang="de",
                                           device=device))

    chunk_count = 10
    chunks = split_dictionary_into_chunks(build_path_to_transcript_dict_mls_german(), split_n=chunk_count)
    for index in range(chunk_count):
        datasets.append(prepare_aligner_corpus(transcript_dict=chunks[index],
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, f"mls_german_chunk_{index}"),
                                               lang="de",
                                               device=device))

    # FRENCH

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10fr,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_French"),
                                           lang="fr",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_mls_french,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_french"),
                                           lang="fr",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_blizzard2023_ad_silence_removed,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "ad_e"),
                                           lang="fr",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_blizzard2023_neb_silence_removed,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "neb"),
                                           lang="fr",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_blizzard2023_neb_e_silence_removed,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "neb_e"),
                                           lang="fr",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_synpaflex_norm_subset,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "synpaflex"),
                                           lang="fr",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_siwis_subset,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "siwis"),
                                           lang="fr",
                                           device=device))

    # SPANISH

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_mls_spanish,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_spanish"),
                                           lang="es",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10es,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Spanish"),
                                           lang="es",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_spanish_blizzard_train,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "spanish_blizzard"),
                                           lang="es",
                                           device=device))

    # CHINESE

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10cmn,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_chinese"),
                                           lang="cmn",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_aishell3,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "aishell3"),
                                           lang="cmn",
                                           device=device))

    # PORTUGUESE

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_mls_portuguese,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_porto"),
                                           lang="pt-br",
                                           device=device))

    # POLISH

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_mls_polish,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_polish"),
                                           lang="pl",
                                           device=device))

    # ITALIAN

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_mls_italian,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_italian"),
                                           lang="it",
                                           device=device))

    # DUTCH

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_mls_dutch,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_dutch"),
                                           lang="nl",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10nl,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Dutch"),
                                           lang="nl",
                                           device=device))

    # GREEK

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10el,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Greek"),
                                           lang="el",
                                           device=device))

    # FINNISH

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10fi,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Finnish"),
                                           lang="fi",
                                           device=device))

    # VIETNAMESE

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_VIVOS_viet,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "VIVOS_viet"),
                                           lang="vi",
                                           device=device))

    # RUSSIAN

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10ru,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Russian"),
                                           lang="ru",
                                           device=device))

    # HUNGARIAN

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10hu,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Hungarian"),
                                           lang="hu",
                                           device=device))

    train_set = ConcatDataset(datasets)
    save_dir = os.path.join(MODELS_DIR, "Aligner")
    os.makedirs(save_dir, exist_ok=True)
    save_dir_aligner = save_dir + "/aligner"
    os.makedirs(save_dir_aligner, exist_ok=True)

    train_aligner(train_dataset=train_set,
                  device=device,
                  save_directory=save_dir,
                  steps=1500000,
                  batch_size=32,
                  path_to_checkpoint=resume_checkpoint,
                  fine_tune=finetune,
                  debug_img_path=save_dir_aligner,
                  resume=resume)
