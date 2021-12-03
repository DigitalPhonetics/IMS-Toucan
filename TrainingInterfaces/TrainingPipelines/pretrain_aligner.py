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

    cache_dir_english = os.path.join("Corpora", "Nancy")
    os.makedirs(cache_dir_english, exist_ok=True)
    languages.append("en")

    cache_dir_greek = os.path.join("Corpora", "meta_Greek")
    os.makedirs(cache_dir_greek, exist_ok=True)
    languages.append("el")

    cache_dir_spanish = os.path.join("Corpora", "meta_Spanish")
    os.makedirs(cache_dir_spanish, exist_ok=True)
    languages.append("es")

    cache_dir_finnish = os.path.join("Corpora", "meta_Finnish")
    os.makedirs(cache_dir_finnish, exist_ok=True)
    languages.append("fi")

    cache_dir_russian = os.path.join("Corpora", "meta_Russian")
    os.makedirs(cache_dir_russian, exist_ok=True)
    languages.append("ru")

    cache_dir_hungarian = os.path.join("Corpora", "meta_Hungarian")
    os.makedirs(cache_dir_hungarian, exist_ok=True)
    languages.append("hu")

    cache_dir_dutch = os.path.join("Corpora", "meta_Dutch")
    os.makedirs(cache_dir_dutch, exist_ok=True)
    languages.append("nl")

    cache_dir_french = os.path.join("Corpora", "meta_French")
    os.makedirs(cache_dir_french, exist_ok=True)
    languages.append("fr")

    cache_dir_german = os.path.join("Corpora", "meta_German")
    os.makedirs(cache_dir_german, exist_ok=True)
    languages.append("de")

    datasets.append(AlignerDataset(build_path_to_transcript_dict_nancy(),
                                   cache_dir=cache_dir_english,
                                   lang="en"))

    datasets.append(AlignerDataset(build_path_to_transcript_dict_css10el(),
                                   cache_dir=cache_dir_greek,
                                   lang="el"))

    datasets.append(AlignerDataset(build_path_to_transcript_dict_css10es(),
                                   cache_dir=cache_dir_spanish,
                                   lang="es"))

    datasets.append(AlignerDataset(build_path_to_transcript_dict_css10fi(),
                                   cache_dir=cache_dir_finnish,
                                   lang="fi"))

    datasets.append(AlignerDataset(build_path_to_transcript_dict_css10ru(),
                                   cache_dir=cache_dir_russian,
                                   lang="ru"))

    datasets.append(AlignerDataset(build_path_to_transcript_dict_css10hu(),
                                   cache_dir=cache_dir_hungarian,
                                   lang="hu"))

    datasets.append(AlignerDataset(build_path_to_transcript_dict_css10nl(),
                                   cache_dir=cache_dir_dutch,
                                   lang="nl"))

    datasets.append(AlignerDataset(build_path_to_transcript_dict_css10fr(),
                                   cache_dir=cache_dir_french,
                                   lang="fr"))

    datasets.append(AlignerDataset(build_path_to_transcript_dict_karlsson(),
                                   cache_dir=cache_dir_german,
                                   lang="de"))

    train_set = ConcatDataset(datasets)
    save_dir = os.path.join("Models", "Aligner")
    os.makedirs(save_dir, exist_ok=True)
    save_dir_aligner = save_dir+"/aligner"
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
