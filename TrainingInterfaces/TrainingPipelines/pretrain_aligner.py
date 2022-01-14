import random

import torch
from torch.utils.data import ConcatDataset

from TrainingInterfaces.Text_to_Spectrogram.AutoAligner.AlignerDataset import AlignerDataset
from TrainingInterfaces.Text_to_Spectrogram.AutoAligner.autoaligner_train_loop import train_loop as train_aligner
from Utility.path_to_transcript_dicts import *


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume):
    if gpu_id == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")

    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
        device = torch.device("cuda")

    torch.manual_seed(131714)
    random.seed(131714)
    torch.random.manual_seed(131714)

    print("Preparing")

    languages = list()
    datasets = list()

    languages.append("fr")
    datasets.append(AlignerDataset(build_path_to_transcript_dict_att_hack(),
                                   cache_dir=os.path.join("Corpora", "expressive_French"),
                                   lang="fr"))

    languages.append("en")
    datasets.append(AlignerDataset(build_path_to_transcript_dict_vctk(),
                                   cache_dir=os.path.join("Corpora", "vctk"),
                                   lang="en"))

    languages.append("de")
    datasets.append(AlignerDataset(build_path_to_transcript_dict_thorsten(),
                                   cache_dir=os.path.join("Corpora", "Thorsten"),
                                   lang="de"))

    languages.append("el")
    datasets.append(AlignerDataset(build_path_to_transcript_dict_css10el(),
                                   cache_dir=os.path.join("Corpora", "meta_Greek"),
                                   lang="el"))

    languages.append("es")
    datasets.append(AlignerDataset(build_path_to_transcript_dict_css10es(),
                                   cache_dir=os.path.join("Corpora", "meta_Spanish"),
                                   lang="es"))

    languages.append("fi")
    datasets.append(AlignerDataset(build_path_to_transcript_dict_css10fi(),
                                   cache_dir=os.path.join("Corpora", "meta_Finnish"),
                                   lang="fi"))

    languages.append("ru")
    datasets.append(AlignerDataset(build_path_to_transcript_dict_css10ru(),
                                   cache_dir=os.path.join("Corpora", "meta_Russian"),
                                   lang="ru"))

    languages.append("hu")
    datasets.append(AlignerDataset(build_path_to_transcript_dict_css10hu(),
                                   cache_dir=os.path.join("Corpora", "meta_Hungarian"),
                                   lang="hu"))

    languages.append("nl")
    datasets.append(AlignerDataset(build_path_to_transcript_dict_css10nl(),
                                   cache_dir=os.path.join("Corpora", "meta_Dutch"),
                                   lang="nl"))

    languages.append("fr")
    datasets.append(AlignerDataset(build_path_to_transcript_dict_css10fr(),
                                   cache_dir=os.path.join("Corpora", "meta_French"),
                                   lang="fr"))

    languages.append("de")
    datasets.append(AlignerDataset(build_path_to_transcript_dict_karlsson(),
                                   cache_dir=os.path.join("Corpora", "Karlsson"),
                                   lang="de"))

    languages.append("en")
    datasets.append(AlignerDataset(build_path_to_transcript_dict_nancy(),
                                   cache_dir=os.path.join("Corpora", "Nancy"),
                                   lang="en"))

    languages.append("en")
    datasets.append(AlignerDataset(build_path_to_transcript_dict_ljspeech(),
                                   cache_dir=os.path.join("Corpora", "LJSpeech"),
                                   lang="en"))

    languages.append("en")
    datasets.append(AlignerDataset(build_path_to_transcript_dict_libritts(),
                                   cache_dir=os.path.join("Corpora", "libri"),
                                   lang="en"))

    languages.append("en")
    datasets.append(AlignerDataset(build_path_to_transcript_dict_nvidia_hifitts(),
                                   cache_dir=os.path.join("Corpora", "hifiTTS"),
                                   lang="en"))

    languages.append("de")
    datasets.append(AlignerDataset(build_path_to_transcript_dict_hokuspokus(),
                                   cache_dir=os.path.join("Corpora", "Hokus"),
                                   lang="de"))

    train_set = ConcatDataset(datasets)
    save_dir = os.path.join("Models", "Aligner")
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
