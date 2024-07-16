"""

STAGE 3: Training on as much data as possible

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

    base_dir = os.path.join(MODELS_DIR, "ToucanTTS_MassiveDataBigModel_stage2_reworked_v4")
    if model_dir is not None:
        meta_save_dir = model_dir
    else:
        meta_save_dir = base_dir
    os.makedirs(meta_save_dir, exist_ok=True)

    print("Preparing")

    if gpu_count > 1:
        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(backend="nccl", world_size=gpu_count, rank=rank)
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
    chunk_count = 100
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

    lang_to_datasets["eng"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_vctk,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "vctk"),
                                                      lang="eng",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["eng"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_nvidia_hifitts,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "hifi"),
                                                      lang="eng",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["eng"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_CREMA_D,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "cremad"),
                                                      lang="eng",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["eng"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_EmoV_DB,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "emovdb"),
                                                      lang="eng",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["eng"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_RAVDESS,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "ravdess"),
                                                      lang="eng",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["eng"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_ESDS,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "esds"),
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

    lang_to_datasets["eng"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_ears,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "ears"),
                                                      lang="eng",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    # GERMAN
    lang_to_datasets["deu"] = list()

    lang_to_datasets["deu"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_karlsson,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "Karlsson"),
                                                      lang="deu",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["deu"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_eva,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "Eva"),
                                                      lang="deu",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["deu"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_hokus,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "Hokus"),
                                                      lang="deu",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["deu"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_bernd,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "Bernd"),
                                                      lang="deu",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["deu"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_friedrich,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "Friedrich"),
                                                      lang="deu",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["deu"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_hui_others,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "hui_others"),
                                                      lang="deu",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["deu"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_thorsten_emotional(),
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "thorsten_emotional"),
                                                      lang="deu",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["deu"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_thorsten_neutral(),
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "thorsten_neutral"),
                                                      lang="deu",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["deu"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_thorsten_2022_10(),
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "thorsten_2022"),
                                                      lang="deu",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    chunk_count = 20
    chunks = split_dictionary_into_chunks(build_path_to_transcript_dict_mls_german(), split_n=chunk_count)
    for index in range(chunk_count):
        lang_to_datasets["deu"].append(prepare_tts_corpus(transcript_dict=chunks[index],
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, f"mls_german_chunk_{index}"),
                                                          lang="deu",
                                                          gpu_count=gpu_count,
                                                          rank=rank))

    # FRENCH

    lang_to_datasets["fra"] = list()

    lang_to_datasets["fra"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_css10fr,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_French"),
                                                      lang="fra",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["fra"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_mls_french,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_french"),
                                                      lang="fra",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["fra"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_blizzard2023_ad_silence_removed,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "ad_e"),
                                                      lang="fra",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["fra"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_blizzard2023_neb_silence_removed,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "neb"),
                                                      lang="fra",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["fra"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_blizzard2023_neb_e_silence_removed,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "neb_e"),
                                                      lang="fra",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["fra"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_synpaflex_norm_subset,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "synpaflex"),
                                                      lang="fra",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["fra"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_siwis_subset,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "siwis"),
                                                      lang="fra",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    # SPANISH

    lang_to_datasets["spa"] = list()

    lang_to_datasets["spa"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_mls_spanish,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_spanish"),
                                                      lang="spa",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["spa"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_css10es,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Spanish"),
                                                      lang="spa",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["spa"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_spanish_blizzard_train,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "spanish_blizzard"),
                                                      lang="spa",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    # CHINESE

    lang_to_datasets["cmn"] = list()

    lang_to_datasets["cmn"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_css10cmn,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_chinese"),
                                                      lang="cmn",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["cmn"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_aishell3,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "aishell3"),
                                                      lang="cmn",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    # PORTUGUESE

    lang_to_datasets["por"] = list()

    lang_to_datasets["por"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_mls_portuguese,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_porto"),
                                                      lang="por",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    # POLISH

    lang_to_datasets["pol"] = list()

    lang_to_datasets["pol"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_mls_polish,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_polish"),
                                                      lang="pol",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    # ITALIAN

    lang_to_datasets["ita"] = list()

    lang_to_datasets["ita"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_mls_italian,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_italian"),
                                                      lang="ita",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    # DUTCH

    lang_to_datasets["nld"] = list()

    lang_to_datasets["nld"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_mls_dutch,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_dutch"),
                                                      lang="nld",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["nld"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_css10nl,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Dutch"),
                                                      lang="nld",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    # GREEK

    lang_to_datasets["ell"] = list()

    lang_to_datasets["ell"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_css10el,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Greek"),
                                                      lang="ell",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    # FINNISH

    lang_to_datasets["fin"] = list()

    lang_to_datasets["fin"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_css10fi,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Finnish"),
                                                      lang="fin",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    # VIETNAMESE

    lang_to_datasets["vie"] = list()

    lang_to_datasets["vie"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_VIVOS_viet,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "VIVOS_viet"),
                                                      lang="vie",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    # Japanese

    lang_to_datasets["jpn"] = list()

    lang_to_datasets["jpn"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_captain_japanese,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "captain_japanese"),
                                                      lang="jpn",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["jpn"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_jvs,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "jvs"),
                                                      lang="jpn",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    # RUSSIAN

    lang_to_datasets["rus"] = list()

    lang_to_datasets["rus"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_css10ru,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Russian"),
                                                      lang="rus",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    # HUNGARIAN

    lang_to_datasets["hun"] = list()

    lang_to_datasets["hun"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_css10hu,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Hungarian"),
                                                      lang="hun",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    # DIVERSE INDIC LANGUAGES

    lang_to_datasets["asm"] = list()

    lang_to_datasets["asm"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_indicvoices_Assamese,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "indic_assamese"),
                                                      lang="asm",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["ben"] = list()

    lang_to_datasets["ben"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_indicvoices_Bengali,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "indic_bengali"),
                                                      lang="ben",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["brx"] = list()

    lang_to_datasets["brx"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_indicvoices_Bodo,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "indic_bodo"),
                                                      lang="brx",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["dgo"] = list()

    lang_to_datasets["dgo"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_indicvoices_Dogri,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "indic_dogri"),
                                                      lang="dgo",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["guj"] = list()

    lang_to_datasets["guj"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_indicvoices_Gujarati,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "indic_gujarati"),
                                                      lang="guj",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["hin"] = list()

    lang_to_datasets["hin"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_indicvoices_Hindi,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "indic_hindi"),
                                                      lang="hin",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["kan"] = list()

    lang_to_datasets["kan"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_indicvoices_Kannada,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "indic_kannada"),
                                                      lang="kan",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["kas"] = list()

    lang_to_datasets["kas"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_indicvoices_Kashmiri,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "indic_kashmiri"),
                                                      lang="kas",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["knn"] = list()

    lang_to_datasets["knn"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_indicvoices_Konkani,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "indic_konkani"),
                                                      lang="knn",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["mai"] = list()

    lang_to_datasets["mai"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_indicvoices_Maithili,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "indic_maithili"),
                                                      lang="mai",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["mal"] = list()

    lang_to_datasets["mal"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_indicvoices_Malayalam,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "indic_malayalam"),
                                                      lang="mal",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["mni"] = list()

    lang_to_datasets["mni"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_indicvoices_Manipuri,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "indic_manipuri"),
                                                      lang="mni",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["mar"] = list()

    lang_to_datasets["mar"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_indicvoices_Marathi,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "indic_marathi"),
                                                      lang="mar",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["nep"] = list()

    lang_to_datasets["nep"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_indicvoices_Nepali,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "indic_nepali"),
                                                      lang="nep",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["ory"] = list()

    lang_to_datasets["ory"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_indicvoices_Odia,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "indic_odia"),
                                                      lang="ory",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["pan"] = list()

    lang_to_datasets["pan"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_indicvoices_Punjabi,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "indic_punjabi"),
                                                      lang="pan",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["san"] = list()

    lang_to_datasets["san"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_indicvoices_Sanskrit,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "indic_sanskrit"),
                                                      lang="san",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["sat"] = list()

    lang_to_datasets["sat"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_indicvoices_Santali,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "indic_santali"),
                                                      lang="sat",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["snd"] = list()

    lang_to_datasets["snd"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_indicvoices_Sindhi,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "indic_sindhi"),
                                                      lang="snd",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["tam"] = list()

    lang_to_datasets["tam"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_indicvoices_Tamil,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "indic_tamil"),
                                                      lang="tam",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["tel"] = list()

    lang_to_datasets["tel"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_indicvoices_Telugu,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "indic_telugu"),
                                                      lang="tel",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    lang_to_datasets["urd"] = list()

    lang_to_datasets["urd"].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_indicvoices_Urdu,
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "indic_urdu"),
                                                      lang="urd",
                                                      gpu_count=gpu_count,
                                                      rank=rank))

    # DIVERSE

    lang_id = "bem"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_bembaspeech(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "bembaspeech"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "swh"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_alffa_sw(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "alffa_sw"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "amh"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_alffa_am(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "alffa_am"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "wol"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_alffa_wo(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "alffa_wo"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "mal"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_malayalam(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "malayalam"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "mal"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_msc(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "msc"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "chv"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_chuvash(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "chuvash"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "iba"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_iban(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "iban"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))

    lang_id = "jav"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_javanese_speech(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "javanese_speech"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "fon"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_african_voices_fon_alf(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_fon_alf"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "hau"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_african_voices_hausa_cmv(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_hausa_cmv"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "lbb"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_african_voices_ibibio_lst(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_ibibio_lst"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "kik"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_african_voices_kikuyu_opb(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_kikuyu_opb"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "lin"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_african_voices_lingala_opb(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_lingala_opb"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "lug"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_african_voices_ganda_cmv(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_ganda_cmv"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "luo"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_african_voices_luo_afv(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_luo_afv"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "luo"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_african_voices_luo_opb(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_luo_opb"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "swh"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_african_voices_swahili_llsti(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_swahili_llsti"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "sxb"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_african_voices_suba_afv(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_suba_afv"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "wol"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_african_voices_wolof_alf(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_wolof_alf"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "yor"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_african_voices_yoruba_opb(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_yoruba_opb"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "nya"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_zambezi_voice_nyanja(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "zambezi_voice_nyanja"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "loz"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_zambezi_voice_lozi(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "zambezi_voice_lozi"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "toi"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_zambezi_voice_tonga(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "zambezi_voice_tonga"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "afr"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_afrikaans(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_afrikaans"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "amh"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_amharic(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_amharic"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "arb"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_arabic(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_arabic"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "asm"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_assamese(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_assamese"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "ast"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_asturian(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_asturian"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "azj"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_azerbaijani(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_azerbaijani"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "bel"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_belarusian(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_belarusian"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "bul"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_bulgarian(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_bulgarian"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "ben"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_bengali(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_bengali"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "bos"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_bosnian(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_bosnian"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "cat"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_catalan(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_catalan"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "ceb"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_cebuano(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_cebuano"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "sdh"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_sorani_kurdish(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_sorani_kurdish"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "cmn"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_mandarin(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_mandarin"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "ces"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_czech(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_czech"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "cym"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_welsh(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_welsh"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "dan"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_danish(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_danish"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "deu"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_german(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_german"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "ell"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_greek(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_greek"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "eng"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_english(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_english"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "spa"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_spanish(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_spanish"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "ekk"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_estonian(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_estonian"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "pes"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_persian(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_persian"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "fin"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_finnish(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_finnish"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "fil"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_filipino(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_filipino"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "fra"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_french(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_french"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "gle"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_irish(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_irish"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "glg"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_galician(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_galician"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "guj"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_gujarati(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_gujarati"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "hau"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_hausa(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_hausa"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "heb"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_hebrew(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_hebrew"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "hin"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_hindi(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_hindi"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "hrv"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_croatian(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_croatian"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "hun"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_hungarian(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_hungarian"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "hye"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_armenian(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_armenian"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "ind"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_indonesian(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_indonesian"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "ibo"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_igbo(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_igbo"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "isl"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_icelandic(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_icelandic"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "ita"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_italian(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_italian"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "jav"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_javanese(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_javanese"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "kat"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_georgian(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_georgian"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "kam"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_kamba(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_kamba"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "kea"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_kabuverdianu(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_kabuverdianu"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "kaz"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_kazakh(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_kazakh"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "khm"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_khmer(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_khmer"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "kan"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_kannada(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_kannada"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "kor"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_korean(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_korean"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "ltz"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_luxembourgish(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_luxembourgish"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "lug"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_ganda(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_ganda"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "lin"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_lingala(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_lingala"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "lao"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_lao(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_lao"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "lit"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_lithuanian(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_lithuanian"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "luo"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_luo(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_luo"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "lvs"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_latvian(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_latvian"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "mri"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_maori(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_maori"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "mkd"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_macedonian(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_macedonian"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "mal"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_malayalam(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_malayalam"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "xng"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_mongolian(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_mongolian"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "mar"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_marathi(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_marathi"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "zsm"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_malay(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_malay"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "mlt"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_maltese(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_maltese"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "nld"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_dutch(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_dutch"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "nya"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_nyanja(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_nyanja"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "oci"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_occitan(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_occitan"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "ory"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_oriya(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_oriya"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "pan"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_punjabi(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_punjabi"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "pol"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_polish(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_polish"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "pst"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_pashto(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_pashto"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "por"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_portuguese(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_portuguese"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "ron"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_romanian(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_romanian"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "rus"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_russian(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_russian"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "snd"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_sindhi(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_sindhi"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "slk"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_slovak(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_slovak"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "slv"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_slovenian(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_slovenian"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "sna"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_shona(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_shona"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "som"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_somali(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_somali"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "srp"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_serbian(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_serbian"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "swe"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_swedish(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_swedish"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "swh"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_swahili(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_swahili"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "tam"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_tamil(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_tamil"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "tel"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_telugu(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_telugu"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "tgk"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_tajik(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_tajik"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "tur"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_turkish(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_turkish"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "ukr"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_ukrainian(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_ukrainian"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "umb"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_umbundu(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_umbundu"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "urd"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_urdu(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_urdu"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "uzn"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_uzbek(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_uzbek"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "vie"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_vietnamese(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_vietnamese"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "wol"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_wolof(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_wolof"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "yor"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_yoruba(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_yoruba"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "gle"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_living_audio_dataset_irish(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "living_audio_dataset_irish"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "nld"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_living_audio_dataset_dutch(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "living_audio_dataset_dutch"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "rus"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_living_audio_dataset_russian(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "living_audio_dataset_russian"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "ron"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_romanian_db(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "romanian_db"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "pes"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_shemo(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "shemo"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "eng"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_cmu_arctic(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "cmu_arctic"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "arb"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_clartts(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "clartts"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "bhd"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_snow_mountain_bhadrawahi(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_bhadrawahi"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "kfs"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_snow_mountain_bilaspuri(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_bilaspuri"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "dgo"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_snow_mountain_dogri(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_dogri"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "gbk"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_snow_mountain_gaddi(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_gaddi"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "bgc"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_snow_mountain_haryanvi(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_haryanvi"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "hin"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_snow_mountain_hindi(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_hindi"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "xnr"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_snow_mountain_kangri(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_kangri"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "kan"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_snow_mountain_kannada(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_kannada"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "kfx"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_snow_mountain_kulvi(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_kulvi"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "kfx"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_snow_mountain_kulvi_outer_seraji(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_kulvi_outer_seraji"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "mal"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_snow_mountain_malayalam(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_malayalam"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "mjl"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_snow_mountain_mandeali(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_mandeali"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "bfz"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_snow_mountain_pahari_mahasui(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_pahari_mahasui"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "tam"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_snow_mountain_tamil(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_tamil"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "tel"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_snow_mountain_telugu(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_telugu"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))
    lang_id = "ukr"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_ukrainian_lada(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "ukrainian_lada"),
                                                        lang=lang_id,
                                                        gpu_count=gpu_count,
                                                        rank=rank))

    for lang in ["acf", "bss", "deu", "inb", "nca", "quh", "wap", "acr", "bus", "dgr", "ind", "maz", "nch", "qul", "tav", "wmw", "acu", "byr", "dik", "iou", "mbb", "ncj", "qvc", "tbc", "xed", "agd", "bzh", "djk", "ipi", "mbc", "ncl", "qve", "tbg", "xon", "agg", "bzj", "dop", "jac", "mbh", "ncu", "qvh", "tbl", "xtd", "agn",
                 "caa", "jic", "mbj", "ndj", "qvm", "tbz", "xtm", "agr", "cab", "emp", "jiv", "mbt", "nfa", "qvn", "tca", "yaa", "agu", "cap", "eng", "jvn", "mca", "ngp", "qvs", "tcs", "yad", "aia", "car", "ese", "mcb", "ngu", "qvw", "yal", "cax", "kaq", "mcd", "nhe", "qvz", "tee", "ycn", "ake", "cbc",
                 "far", "mco", "qwh", "yka", "alp", "cbi", "fra", "kdc", "mcp", "nhu", "qxh", "ame", "cbr", "gai", "kde", "mcq", "nhw", "qxn", "tew", "yre", "amf", "cbs", "gam", "kdl", "mdy", "nhy", "qxo", "tfr", "yva", "amk", "cbt", "geb", "kek", "med", "nin", "rai", "zaa", "apb", "cbu", "glk",
                 "ken", "mee", "nko", "rgu", "zab", "apr", "cbv", "meq", "nld", "tgo", "zac", "arl", "cco", "gng", "kje", "met", "nlg", "rop", "tgp", "zad", "grc", "klv", "mgh", "nnq", "rro", "zai", "ata", "cek", "gub", "kmu", "mib", "noa", "ruf", "tna", "zam", "atb", "cgc", "guh", "kne",
                 "mie", "not", "rug", "tnk", "zao", "atg", "chf", "knf", "mih", "npl", "rus", "tnn", "zar", "awb", "chz", "gum", "knj", "mil", "sab", "tnp", "zas", "cjo", "guo", "ksr", "mio", "obo", "seh", "toc", "zav", "azg", "cle", "gux", "kue", "mit", "omw", "sey", "tos", "zaw", "azz", "cme", "gvc", "kvn", "miz",
                 "ood", "sgb", "tpi", "zca", "bao", "cni", "gwi", "kwd", "mkl", "shp", "tpt", "zga", "bba", "cnl", "gym", "kwf", "mkn", "ote", "sja", "trc", "ziw", "bbb", "cnt", "gyr", "kwi", "mop", "otq", "snn", "ttc", "zlm", "cof", "hat", "kyc", "mox", "pab", "snp", "tte", "zos", "bgt", "con", "kyf", "mpm", "pad",
                 "som", "tue", "zpc", "bjr", "cot", "heb", "kyg", "mpp", "soy", "tuf", "zpl", "bjv", "cpa", "kyq", "mpx", "pao", "spa", "tuo", "zpm", "bjz", "cpb", "hlt", "kyz", "mqb", "pib", "spp", "tur", "zpo", "bkd", "cpu", "hns", "lac", "mqj", "pir", "spy", "txq", "zpu", "blz", "crn", "hto", "lat", "msy", "pjt", "sri",
                 "txu", "zpz", "bmr", "cso", "hub", "lex", "mto", "pls", "srm", "udu", "ztq", "bmu", "ctu", "lgl", "muy", "poi", "srn", "ukr", "zty", "bnp", "cuc", "lid", "mxb", "pol", "stp", "upv", "zyp", "boa", "cui", "huu", "mxq", "por", "sus", "ura", "boj", "cuk", "huv", "llg", "mxt", "poy", "suz", "urb", "box",
                 "cwe", "hvn", "prf", "swe", "urt", "bpr", "cya", "ign", "lww", "myk", "ptu", "swh", "usp", "bps", "daa", "ikk", "maj", "myy", "sxb", "vid", "bqc", "dah", "nab", "qub", "tac", "vie", "bqp", "ded", "imo", "maq", "nas", "quf", "taj", "vmy"]:

        if lang not in lang_to_datasets:
            lang_to_datasets[lang] = list()

        lang_to_datasets[lang].append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_mms_template(lang=lang),
                                                         corpus_dir=os.path.join(PREPROCESSING_DIR, f"mms_{lang}"),
                                                         lang=f"{lang}",
                                                         gpu_count=gpu_count,
                                                         rank=rank))

    for lang in lang_to_datasets:
        datasets.append(ConcatDataset(lang_to_datasets[lang]))
    re_ordered_datasets = list()
    collection_dataset = list()
    for dataset in datasets:
        if len(dataset) < 1000:  # This language is too small to be a task on its own, so we join it with other tiny languages to make a combined task.
            collection_dataset.append(dataset)
        else:
            re_ordered_datasets.append(dataset)
    if len(collection_dataset) != 0:
        re_ordered_datasets.append(ConcatDataset(collection_dataset))
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
    for train_set in re_ordered_datasets:
        train_samplers.append(torch.utils.data.RandomSampler(train_set))

    if use_wandb:
        if rank == 0:
            wandb.init(
                name=f"{__name__.split('.')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}" if wandb_resume_id is None else None,
                id=wandb_resume_id,  # this is None if not specified in the command line arguments.
                resume="must" if wandb_resume_id is not None else None)
    train_loop(net=model,
               batch_size=16,
               warmup_steps=1000,
               device=torch.device("cuda"),
               datasets=re_ordered_datasets,
               save_directory=meta_save_dir,
               path_to_checkpoint=resume_checkpoint,
               resume=resume,
               fine_tune=finetune,
               steps=200000,
               steps_per_checkpoint=1000,
               lr=0.001,
               use_wandb=use_wandb,
               train_samplers=train_samplers,
               gpu_count=gpu_count,
               use_less_loss=False)
    if use_wandb:
        wandb.finish()
