import torch
from torch.utils.data import ConcatDataset

from Modules.Aligner.autoaligner_train_loop import train_loop as train_aligner
from Utility.corpus_preparation import prepare_aligner_corpus
from Utility.path_to_transcript_dicts import *
from Utility.storage_config import MODELS_DIR
from Utility.storage_config import PREPROCESSING_DIR


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume, use_wandb, wandb_resume_id, gpu_count):
    if gpu_id == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    if gpu_count > 1:
        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(backend="nccl")
    else:
        rank = 0

    print("Preparing")

    lang_to_datasets = dict()

    # ENGLISH

    lang_to_datasets["eng"] = list()

    lang_to_datasets["eng"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_nancy,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "Nancy"),
                                                          lang="eng",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))
    chunk_count = 100
    chunks = split_dictionary_into_chunks(build_path_to_transcript_mls_english(), split_n=chunk_count)
    for index in range(chunk_count):
        lang_to_datasets["eng"].append(prepare_aligner_corpus(transcript_dict=chunks[index],
                                                              corpus_dir=os.path.join(PREPROCESSING_DIR, f"mls_english_chunk_{index}"),
                                                              lang="eng",
                                                              gpu_count=gpu_count,
                                                              rank=rank,
                                                              device=device))

    lang_to_datasets["eng"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_ryanspeech,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "Ryan"),
                                                          lang="eng",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    lang_to_datasets["eng"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_ljspeech,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "LJSpeech"),
                                                          lang="eng",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    lang_to_datasets["eng"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_libritts_all_clean,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "libri_all_clean"),
                                                          lang="eng",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    lang_to_datasets["eng"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_vctk,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "vctk"),
                                                          lang="eng",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    lang_to_datasets["eng"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_nvidia_hifitts,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "hifi"),
                                                          lang="eng",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    lang_to_datasets["eng"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_CREMA_D,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "cremad"),
                                                          lang="eng",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    lang_to_datasets["eng"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_EmoV_DB,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "emovdb"),
                                                          lang="eng",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    lang_to_datasets["eng"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_RAVDESS,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "ravdess"),
                                                          lang="eng",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    lang_to_datasets["eng"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_ESDS,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "esds"),
                                                          lang="eng",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    lang_to_datasets["eng"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_blizzard_2013,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "blizzard2013"),
                                                          lang="eng",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    lang_to_datasets["eng"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_jenny,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "jenny"),
                                                          lang="eng",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    lang_to_datasets["eng"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_ears,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "ears"),
                                                          lang="eng",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    # GERMAN
    lang_to_datasets["deu"] = list()

    lang_to_datasets["deu"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_karlsson,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "Karlsson"),
                                                          lang="deu",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    lang_to_datasets["deu"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_eva,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "Eva"),
                                                          lang="deu",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    lang_to_datasets["deu"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_hokus,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "Hokus"),
                                                          lang="deu",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    lang_to_datasets["deu"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_bernd,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "Bernd"),
                                                          lang="deu",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    lang_to_datasets["deu"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_friedrich,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "Friedrich"),
                                                          lang="deu",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    lang_to_datasets["deu"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_hui_others,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "hui_others"),
                                                          lang="deu",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    lang_to_datasets["deu"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_thorsten_emotional(),
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "thorsten_emotional"),
                                                          lang="deu",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    lang_to_datasets["deu"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_thorsten_neutral(),
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "thorsten_neutral"),
                                                          lang="deu",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    lang_to_datasets["deu"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_thorsten_2022_10(),
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "thorsten_2022"),
                                                          lang="deu",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    chunk_count = 20
    chunks = split_dictionary_into_chunks(build_path_to_transcript_mls_german(), split_n=chunk_count)
    for index in range(chunk_count):
        lang_to_datasets["deu"].append(prepare_aligner_corpus(transcript_dict=chunks[index],
                                                              corpus_dir=os.path.join(PREPROCESSING_DIR, f"mls_german_chunk_{index}"),
                                                              lang="deu",
                                                              gpu_count=gpu_count,
                                                              rank=rank,
                                                              device=device))

    # FRENCH

    lang_to_datasets["fra"] = list()

    lang_to_datasets["fra"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_css10fr,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_French"),
                                                          lang="fra",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    lang_to_datasets["fra"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_mls_french,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_french"),
                                                          lang="fra",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    lang_to_datasets["fra"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_blizzard2023_ad_silence_removed,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "ad_e"),
                                                          lang="fra",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    lang_to_datasets["fra"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_blizzard2023_neb_silence_removed,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "neb"),
                                                          lang="fra",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    lang_to_datasets["fra"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_blizzard2023_neb_e_silence_removed,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "neb_e"),
                                                          lang="fra",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    lang_to_datasets["fra"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_synpaflex_norm_subset,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "synpaflex"),
                                                          lang="fra",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    lang_to_datasets["fra"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_siwis_subset,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "siwis"),
                                                          lang="fra",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    # SPANISH

    lang_to_datasets["spa"] = list()

    lang_to_datasets["spa"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_mls_spanish,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_spanish"),
                                                          lang="spa",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    lang_to_datasets["spa"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_css10es,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Spanish"),
                                                          lang="spa",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    lang_to_datasets["spa"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_spanish_blizzard_train,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "spanish_blizzard"),
                                                          lang="spa",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    # CHINESE

    lang_to_datasets["cmn"] = list()

    lang_to_datasets["cmn"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_css10cmn,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_chinese"),
                                                          lang="cmn",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    lang_to_datasets["cmn"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_aishell3,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "aishell3"),
                                                          lang="cmn",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    # PORTUGUESE

    lang_to_datasets["por"] = list()

    lang_to_datasets["por"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_mls_portuguese,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_porto"),
                                                          lang="por",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    # POLISH

    lang_to_datasets["pol"] = list()

    lang_to_datasets["pol"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_mls_polish,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_polish"),
                                                          lang="pol",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    # ITALIAN

    lang_to_datasets["ita"] = list()

    lang_to_datasets["ita"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_mls_italian,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_italian"),
                                                          lang="ita",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    # DUTCH

    lang_to_datasets["nld"] = list()

    lang_to_datasets["nld"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_mls_dutch,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_dutch"),
                                                          lang="nld",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    lang_to_datasets["nld"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_css10nl,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Dutch"),
                                                          lang="nld",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    # GREEK

    lang_to_datasets["ell"] = list()

    lang_to_datasets["ell"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_css10el,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Greek"),
                                                          lang="ell",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    # FINNISH

    lang_to_datasets["fin"] = list()

    lang_to_datasets["fin"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_css10fi,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Finnish"),
                                                          lang="fin",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    # VIETNAMESE

    lang_to_datasets["vie"] = list()

    lang_to_datasets["vie"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_VIVOS_viet,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "VIVOS_viet"),
                                                          lang="vie",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    # JAPANESE

    lang_to_datasets["jpn"] = list()

    lang_to_datasets["jpn"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_captain_japanese,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "captain_japanese"),
                                                          lang="jpn",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    lang_to_datasets["jpn"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_jvs,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "jvs"),
                                                          lang="jpn",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    # RUSSIAN

    lang_to_datasets["rus"] = list()

    lang_to_datasets["rus"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_css10ru,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Russian"),
                                                          lang="rus",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    # HUNGARIAN

    lang_to_datasets["hun"] = list()

    lang_to_datasets["hun"].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_css10hu,
                                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Hungarian"),
                                                          lang="hun",
                                                          gpu_count=gpu_count,
                                                          rank=rank,
                                                          device=device))

    # DIVERSE

    lang_id = "bem"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_bembaspeech(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "bembaspeech"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "swh"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_alffa_sw(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "alffa_sw"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "amh"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_alffa_am(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "alffa_am"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "wol"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_alffa_wo(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "alffa_wo"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "mal"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_malayalam(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "malayalam"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "mal"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_msc(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "msc"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "chv"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_chuvash(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "chuvash"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "iba"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_iban(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "iban"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))

    lang_id = "jav"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_javanese_speech(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "javanese_speech"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "fon"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_african_voices_fon_alf(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_fon_alf"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "hau"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_african_voices_hausa_cmv(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_hausa_cmv"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "lbb"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_african_voices_ibibio_lst(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_ibibio_lst"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "kik"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_african_voices_kikuyu_opb(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_kikuyu_opb"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "lin"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_african_voices_lingala_opb(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_lingala_opb"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "lug"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_african_voices_ganda_cmv(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_ganda_cmv"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "luo"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_african_voices_luo_afv(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_luo_afv"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "luo"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_african_voices_luo_opb(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_luo_opb"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "swh"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_african_voices_swahili_llsti(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_swahili_llsti"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "sxb"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_african_voices_suba_afv(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_suba_afv"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "wol"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_african_voices_wolof_alf(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_wolof_alf"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "yor"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_african_voices_yoruba_opb(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_yoruba_opb"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "nya"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_zambezi_voice_nyanja(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "zambezi_voice_nyanja"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "loz"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_zambezi_voice_lozi(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "zambezi_voice_lozi"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "toi"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_zambezi_voice_tonga(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "zambezi_voice_tonga"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "afr"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_afrikaans(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_afrikaans"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "amh"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_amharic(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_amharic"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "arb"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_arabic(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_arabic"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "asm"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_assamese(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_assamese"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "ast"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_asturian(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_asturian"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "azj"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_azerbaijani(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_azerbaijani"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "bel"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_belarusian(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_belarusian"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "bul"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_bulgarian(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_bulgarian"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "ben"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_bengali(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_bengali"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "bos"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_bosnian(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_bosnian"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "cat"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_catalan(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_catalan"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "ceb"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_cebuano(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_cebuano"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "sdh"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_sorani_kurdish(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_sorani_kurdish"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "cmn"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_mandarin(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_mandarin"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "ces"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_czech(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_czech"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "cym"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_welsh(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_welsh"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "dan"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_danish(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_danish"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "deu"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_german(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_german"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "ell"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_greek(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_greek"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "eng"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_english(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_english"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "spa"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_spanish(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_spanish"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "ekk"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_estonian(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_estonian"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "pes"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_persian(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_persian"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "fin"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_finnish(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_finnish"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "fil"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_filipino(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_filipino"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "fra"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_french(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_french"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "gle"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_irish(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_irish"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "glg"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_galician(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_galician"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "guj"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_gujarati(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_gujarati"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "hau"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_hausa(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_hausa"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "heb"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_hebrew(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_hebrew"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "hin"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_hindi(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_hindi"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "hrv"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_croatian(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_croatian"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "hun"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_hungarian(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_hungarian"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "hye"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_armenian(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_armenian"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "ind"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_indonesian(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_indonesian"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "ibo"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_igbo(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_igbo"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "isl"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_icelandic(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_icelandic"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "ita"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_italian(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_italian"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "jav"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_javanese(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_javanese"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "kat"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_georgian(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_georgian"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "kam"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_kamba(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_kamba"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "kea"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_kabuverdianu(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_kabuverdianu"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "kaz"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_kazakh(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_kazakh"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "khm"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_khmer(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_khmer"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "kan"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_kannada(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_kannada"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "kor"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_korean(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_korean"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "ltz"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_luxembourgish(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_luxembourgish"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "lug"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_ganda(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_ganda"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "lin"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_lingala(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_lingala"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "lao"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_lao(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_lao"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "lit"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_lithuanian(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_lithuanian"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "luo"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_luo(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_luo"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "lvs"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_latvian(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_latvian"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "mri"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_maori(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_maori"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "mkd"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_macedonian(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_macedonian"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "mal"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_malayalam(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_malayalam"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "xng"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_mongolian(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_mongolian"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "mar"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_marathi(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_marathi"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "zsm"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_malay(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_malay"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "mlt"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_maltese(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_maltese"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "nld"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_dutch(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_dutch"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "nya"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_nyanja(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_nyanja"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "oci"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_occitan(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_occitan"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "ory"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_oriya(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_oriya"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "pan"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_punjabi(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_punjabi"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "pol"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_polish(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_polish"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "pst"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_pashto(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_pashto"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "por"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_portuguese(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_portuguese"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "ron"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_romanian(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_romanian"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "rus"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_russian(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_russian"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "snd"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_sindhi(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_sindhi"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "slk"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_slovak(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_slovak"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "slv"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_slovenian(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_slovenian"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "sna"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_shona(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_shona"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "som"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_somali(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_somali"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "srp"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_serbian(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_serbian"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "swe"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_swedish(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_swedish"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "swh"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_swahili(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_swahili"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "tam"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_tamil(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_tamil"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "tel"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_telugu(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_telugu"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "tgk"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_tajik(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_tajik"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "tur"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_turkish(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_turkish"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "ukr"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_ukrainian(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_ukrainian"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "umb"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_umbundu(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_umbundu"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "urd"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_urdu(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_urdu"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "uzn"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_uzbek(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_uzbek"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "vie"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_vietnamese(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_vietnamese"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "wol"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_wolof(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_wolof"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "yor"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_fleurs_yoruba(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_yoruba"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "gle"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_living_audio_dataset_irish(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "living_audio_dataset_irish"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "nld"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_living_audio_dataset_dutch(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "living_audio_dataset_dutch"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "rus"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_living_audio_dataset_russian(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "living_audio_dataset_russian"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "ron"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_romanian_db(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "romanian_db"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "pes"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_shemo(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "shemo"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "eng"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_cmu_arctic(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "cmu_arctic"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "arb"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_clartts(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "clartts"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "bhd"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_snow_mountain_bhadrawahi(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_bhadrawahi"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "kfs"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_snow_mountain_bilaspuri(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_bilaspuri"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "dgo"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_snow_mountain_dogri(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_dogri"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "gbk"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_snow_mountain_gaddi(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_gaddi"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "bgc"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_snow_mountain_haryanvi(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_haryanvi"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "hin"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_snow_mountain_hindi(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_hindi"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "xnr"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_snow_mountain_kangri(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_kangri"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "kan"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_snow_mountain_kannada(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_kannada"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "kfx"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_snow_mountain_kulvi(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_kulvi"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "kfx"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_snow_mountain_kulvi_outer_seraji(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_kulvi_outer_seraji"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "mal"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_snow_mountain_malayalam(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_malayalam"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "mjl"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_snow_mountain_mandeali(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_mandeali"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "bfz"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_snow_mountain_pahari_mahasui(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_pahari_mahasui"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "tam"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_snow_mountain_tamil(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_tamil"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "tel"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_snow_mountain_telugu(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_telugu"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))
    lang_id = "ukr"
    if lang_id not in lang_to_datasets:
        lang_to_datasets[lang_id] = list()
    lang_to_datasets[lang_id].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_ukrainian_lada(),
                                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "ukrainian_lada"),
                                                            lang=lang_id,
                                                            gpu_count=gpu_count,
                                                            rank=rank,
                                                            device=device))

    for lang in ["acf", "bss", "deu", "inb", "nca", "quh", "wap", "acr", "bus", "dgr", "ind", "maz",
                 "nch", "qul", "tav", "wmw", "acu", "byr", "dik", "iou", "mbb", "ncj", "qvc", "tbc",
                 "xed", "agd", "bzh", "djk", "ipi", "mbc", "ncl", "qve", "tbg", "xon", "agg", "bzj",
                 "dop", "jac", "mbh", "ncu", "qvh", "tbl", "xtd", "agn", "caa", "jic", "mbj", "ndj",
                 "qvm", "tbz", "xtm", "agr", "cab", "emp", "jiv", "mbt", "nfa", "qvn", "tca", "yaa",
                 "agu", "cap", "eng", "jvn", "mca", "ngp", "qvs", "tcs", "yad", "aia", "car", "ese",
                 "mcb", "ngu", "qvw", "yal", "cax", "kaq", "mcd", "nhe", "qvz", "tee", "ycn", "ake",
                 "cbc", "far", "mco", "qwh", "yka", "alp", "cbi", "fra", "kdc", "mcp", "nhu", "qxh",
                 "ame", "cbr", "gai", "kde", "mcq", "nhw", "qxn", "tew", "yre", "amf", "cbs", "gam",
                 "kdl", "mdy", "nhy", "qxo", "tfr", "yva", "amk", "cbt", "geb", "kek", "med", "nin",
                 "rai", "zaa", "apb", "cbu", "glk", "ken", "mee", "nko", "rgu", "zab", "apr", "cbv",
                 "meq", "nld", "tgo", "zac", "arl", "cco", "gng", "kje", "met", "nlg", "rop", "tgp",
                 "zad", "grc", "klv", "mgh", "nnq", "rro", "zai", "ata", "cek", "gub", "kmu", "mib",
                 "noa", "ruf", "tna", "zam", "atb", "cgc", "guh", "kne", "mie", "not", "rug", "tnk",
                 "zao", "atg", "chf", "knf", "mih", "npl", "rus", "tnn", "zar", "awb", "chz", "gum",
                 "knj", "mil", "sab", "tnp", "zas", "cjo", "guo", "ksr", "mio", "obo", "seh", "toc",
                 "zav", "azg", "cle", "gux", "kue", "mit", "omw", "sey", "tos", "zaw", "azz", "cme",
                 "gvc", "kvn", "miz", "ood", "sgb", "tpi", "zca", "bao", "cni", "gwi", "kwd", "mkl",
                 "shp", "tpt", "zga", "bba", "cnl", "gym", "kwf", "mkn", "ote", "sja", "trc", "ziw",
                 "bbb", "cnt", "gyr", "kwi", "mop", "otq", "snn", "ttc", "zlm", "cof", "hat", "kyc",
                 "mox", "pab", "snp", "tte", "zos", "bgt", "con", "kyf", "mpm", "pad", "som", "tue",
                 "zpc", "bjr", "cot", "heb", "kyg", "mpp", "soy", "tuf", "zpl", "bjv", "cpa", "kyq",
                 "mpx", "pao", "spa", "tuo", "zpm", "bjz", "cpb", "hlt", "kyz", "mqb", "pib", "spp",
                 "tur", "zpo", "bkd", "cpu", "hns", "lac", "mqj", "pir", "spy", "txq", "zpu", "blz",
                 "crn", "hto", "lat", "msy", "pjt", "sri", "txu", "zpz", "bmr", "cso", "hub", "lex",
                 "mto", "pls", "srm", "udu", "ztq", "bmu", "ctu", "lgl", "muy", "poi", "srn", "ukr",
                 "zty", "bnp", "cuc", "lid", "mxb", "pol", "stp", "upv", "zyp", "boa", "cui", "huu",
                 "mxq", "por", "sus", "ura", "boj", "cuk", "huv", "llg", "mxt", "poy", "suz", "urb",
                 "box", "cwe", "hvn", "prf", "swe", "urt", "bpr", "cya", "ign", "lww", "myk", "ptu",
                 "swh", "usp", "bps", "daa", "ikk", "maj", "myy", "sxb", "vid", "bqc", "dah", "nab",
                 "qub", "tac", "vie", "bqp", "ded", "imo", "maq", "nas", "quf", "taj", "vmy"]:

        if lang not in lang_to_datasets:
            lang_to_datasets[lang] = list()

        lang_to_datasets[lang].append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_mms_template(lang=lang),
                                                             corpus_dir=os.path.join(PREPROCESSING_DIR, f"mms_{lang}"),
                                                             lang=f"{lang}",
                                                             gpu_count=gpu_count,
                                                             rank=rank,
                                                             device=device))

    datasets = list()
    for lang in lang_to_datasets:
        for dataset in lang_to_datasets[lang]:
            datasets.append(dataset)

    train_set = ConcatDataset(datasets)
    save_dir = os.path.join(MODELS_DIR, "Aligner")
    os.makedirs(save_dir, exist_ok=True)
    save_dir_aligner = save_dir + "/aligner"
    os.makedirs(save_dir_aligner, exist_ok=True)

    train_aligner(train_dataset=train_set,
                  device=device,
                  save_directory=save_dir,
                  steps=1500000,
                  batch_size=16,
                  path_to_checkpoint=resume_checkpoint,
                  fine_tune=finetune,
                  debug_img_path=save_dir_aligner,
                  resume=resume,
                  gpu_count=gpu_count,
                  rank=rank,
                  steps_per_checkpoint=1000)
