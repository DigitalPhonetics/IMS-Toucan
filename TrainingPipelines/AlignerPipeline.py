from torch.utils.data import ConcatDataset

from Architectures.Aligner.autoaligner_train_loop import train_loop as train_aligner
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

    datasets = list()

    # ENGLISH

    chunk_count = 50
    chunks = split_dictionary_into_chunks(build_path_to_transcript_dict_mls_english(), split_n=chunk_count)
    for index in range(chunk_count):
        datasets.append(prepare_aligner_corpus(transcript_dict=chunks[index],
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, f"mls_english_chunk_{index}"),
                                               lang="eng",
                                               device=device,
                                               gpu_count=gpu_count,
                                               rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_nancy,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "Nancy"),
                                           lang="eng",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_elizabeth,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "Elizabeth"),
                                           lang="eng",  # technically, she's british english, not american english
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_ryanspeech,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "Ryan"),
                                           lang="eng",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_ljspeech,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "LJSpeech"),
                                           lang="eng",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_libritts_all_clean,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "libri_all_clean"),
                                           lang="eng",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_vctk,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "vctk"),
                                           lang="eng",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_nvidia_hifitts,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "hifi"),
                                           lang="eng",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_CREMA_D,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "cremad"),
                                           lang="eng",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_EmoV_DB,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "emovdb"),
                                           lang="eng",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_RAVDESS,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "ravdess"),
                                           lang="eng",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_ESDS,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "esds"),
                                           lang="eng",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_blizzard_2013,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "blizzard2013"),
                                           lang="eng",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_jenny,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "jenny"),
                                           lang="eng",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    chunk_count = 30
    chunks = split_dictionary_into_chunks(build_path_to_transcript_dict_gigaspeech(), split_n=chunk_count)
    for index in range(chunk_count):
        datasets.append(prepare_aligner_corpus(transcript_dict=chunks[index],
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, f"gigaspeech_chunk_{index}"),
                                               lang="eng",
                                               device=device,
                                               gpu_count=gpu_count,
                                               rank=rank))

    # GERMAN

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_karlsson,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "Karlsson"),
                                           lang="deu",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_eva,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "Eva"),
                                           lang="deu",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_hokus,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "Hokus"),
                                           lang="deu",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_bernd,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "Bernd"),
                                           lang="deu",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_friedrich,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "Friedrich"),
                                           lang="deu",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_hui_others,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "hui_others"),
                                           lang="deu",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_thorsten_emotional(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "thorsten_emotional"),
                                           lang="deu",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_thorsten_neutral(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "thorsten_neutral"),
                                           lang="deu",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_thorsten_2022_10(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "thorsten_2022"),
                                           lang="deu",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    chunk_count = 10
    chunks = split_dictionary_into_chunks(build_path_to_transcript_dict_mls_german(), split_n=chunk_count)
    for index in range(chunk_count):
        datasets.append(prepare_aligner_corpus(transcript_dict=chunks[index],
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, f"mls_german_chunk_{index}"),
                                               lang="deu",
                                               device=device,
                                               gpu_count=gpu_count,
                                               rank=rank))

    # FRENCH

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10fr,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_French"),
                                           lang="fra",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_mls_french,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_french"),
                                           lang="fra",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_blizzard2023_ad_silence_removed,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "ad_e"),
                                           lang="fra",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_blizzard2023_neb_silence_removed,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "neb"),
                                           lang="fra",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_blizzard2023_neb_e_silence_removed,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "neb_e"),
                                           lang="fra",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_synpaflex_norm_subset,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "synpaflex"),
                                           lang="fra",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_siwis_subset,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "siwis"),
                                           lang="fra",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    # SPANISH

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_mls_spanish,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_spanish"),
                                           lang="spa",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10es,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Spanish"),
                                           lang="spa",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_spanish_blizzard_train,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "spanish_blizzard"),
                                           lang="spa",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    # CHINESE

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10cmn,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_chinese"),
                                           lang="zho",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_aishell3,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "aishell3"),
                                           lang="zho",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    # PORTUGUESE

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_mls_portuguese,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_porto"),
                                           lang="por",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    # POLISH

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_mls_polish,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_polish"),
                                           lang="pol",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    # ITALIAN

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_mls_italian,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_italian"),
                                           lang="ita",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    # DUTCH

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_mls_dutch,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_dutch"),
                                           lang="nld",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10nl,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Dutch"),
                                           lang="nld",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    # GREEK

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10el,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Greek"),
                                           lang="ell",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    # FINNISH

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10fi,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Finnish"),
                                           lang="fin",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    # VIETNAMESE

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_VIVOS_viet,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "VIVOS_viet"),
                                           lang="vie",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    # RUSSIAN

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10ru,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Russian"),
                                           lang="rus",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    # HUNGARIAN

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10hu,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Hungarian"),
                                           lang="hun",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

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
                  steps_per_checkpoint=5000)
