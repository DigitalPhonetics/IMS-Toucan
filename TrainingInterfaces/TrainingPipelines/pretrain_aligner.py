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

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_mls_portuguese(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_porto"),
                                           lang="pt-bt",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_mls_polish(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_polish"),
                                           lang="pl",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_mls_spanish(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_spanish"),
                                           lang="es",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_mls_french(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_french"),
                                           lang="fr",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_mls_italian(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_italian"),
                                           lang="it",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_mls_dutch(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_dutch"),
                                           lang="nl",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_nancy(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "Nancy"),
                                           lang="en",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_karlsson(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "Karlsson"),
                                           lang="de",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10el(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "meta_Greek"),
                                           lang="el",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10es(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "meta_Spanish"),
                                           lang="es",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10fi(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "meta_Finnish"),
                                           lang="fi",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10ru(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "meta_Russian"),
                                           lang="ru",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10hu(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "meta_Hungarian"),
                                           lang="hu",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10nl(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "meta_Dutch"),
                                           lang="nl",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10fr(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "meta_French"),
                                           lang="fr",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_ljspeech(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "LJSpeech"),
                                           lang="en",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_libritts(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "libri"),
                                           lang="en",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_vctk(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "vctk"),
                                           lang="en",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_nvidia_hifitts(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "hifi"),
                                           lang="en",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_spanish_blizzard_train(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "spanish_blizzard"),
                                           lang="es",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_eva(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "Eva"),
                                           lang="de",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_hokus(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "Hokus"),
                                           lang="de",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_bernd(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "Bernd"),
                                           lang="de",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_hui_others(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "hui_others"),
                                           lang="de",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_thorsten(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "Thorsten"),
                                           lang="de",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fluxsing(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "flux_sing"),
                                           lang="en",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10cmn(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_chinese"),
                                           lang="cmn",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_aishell3(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "aishell3"),
                                           lang="cmn",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_VIVOS_viet(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "VIVOS_viet"),
                                           lang="vi",
                                           device=device))

    train_set = ConcatDataset(datasets)
    save_dir = os.path.join(MODELS_DIR, "Aligner")
    os.makedirs(save_dir, exist_ok=True)
    save_dir_aligner = save_dir + "/aligner"
    os.makedirs(save_dir_aligner, exist_ok=True)

    train_aligner(train_dataset=train_set,
                  device=device,
                  save_directory=save_dir,
                  steps=500000,
                  batch_size=32,
                  path_to_checkpoint=resume_checkpoint,
                  fine_tune=finetune,
                  debug_img_path=save_dir_aligner,
                  resume=resume)
