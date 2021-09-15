import random

import torch
from torch.utils.data import ConcatDataset

from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.Tacotron2 import Tacotron2
from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.TacotronDataset import TacotronDataset
from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.tacotron2_train_loop import train_loop
from Utility.path_to_transcript_dicts import *


def run(gpu_id, resume_checkpoint, finetune, model_dir):
    if gpu_id == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")

    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
        device = torch.device("cuda")

    torch.manual_seed(13)
    random.seed(13)

    print("Preparing")
    cache_dir_english = os.path.join("Corpora", "multiling_English")
    os.makedirs(cache_dir_english, exist_ok=True)

    cache_dir_greek = os.path.join("Corpora", "multiling_Greek")
    os.makedirs(cache_dir_greek, exist_ok=True)

    cache_dir_spanish = os.path.join("Corpora", "multiling_Spanish")
    os.makedirs(cache_dir_spanish, exist_ok=True)

    cache_dir_finnish = os.path.join("Corpora", "multiling_Finnish")
    os.makedirs(cache_dir_finnish, exist_ok=True)

    cache_dir_russian = os.path.join("Corpora", "multiling_Russian")
    os.makedirs(cache_dir_russian, exist_ok=True)

    cache_dir_hungarian = os.path.join("Corpora", "multiling_Hungarian")
    os.makedirs(cache_dir_hungarian, exist_ok=True)

    cache_dir_dutch = os.path.join("Corpora", "multiling_Dutch")
    os.makedirs(cache_dir_dutch, exist_ok=True)

    cache_dir_french = os.path.join("Corpora", "multiling_French")
    os.makedirs(cache_dir_french, exist_ok=True)

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join("Models", "Tacotron2_MetaCheckpoint")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    datasets = list()

    datasets.append(TacotronDataset(build_path_to_transcript_dict_nancy(),
                                    cache_dir=cache_dir_english,
                                    lang="en",
                                    speaker_embedding=True,
                                    cut_silences=False,
                                    return_language_id=True))

    datasets.append(TacotronDataset(build_path_to_transcript_dict_css10el(),
                                    cache_dir=cache_dir_greek,
                                    lang="el",
                                    speaker_embedding=True,
                                    cut_silences=False,
                                    return_language_id=True))

    datasets.append(TacotronDataset(build_path_to_transcript_dict_css10es(),
                                    cache_dir=cache_dir_spanish,
                                    lang="es",
                                    speaker_embedding=True,
                                    cut_silences=False,
                                    return_language_id=True))

    datasets.append(TacotronDataset(build_path_to_transcript_dict_css10fi(),
                                    cache_dir=cache_dir_finnish,
                                    lang="fi",
                                    speaker_embedding=True,
                                    cut_silences=False,
                                    return_language_id=True))

    datasets.append(TacotronDataset(build_path_to_transcript_dict_css10ru(),
                                    cache_dir=cache_dir_russian,
                                    lang="ru",
                                    speaker_embedding=True,
                                    cut_silences=False,
                                    return_language_id=True))

    datasets.append(TacotronDataset(build_path_to_transcript_dict_css10hu(),
                                    cache_dir=cache_dir_hungarian,
                                    lang="hu",
                                    speaker_embedding=True,
                                    cut_silences=False,
                                    return_language_id=True))

    datasets.append(TacotronDataset(build_path_to_transcript_dict_css10nl(),
                                    cache_dir=cache_dir_dutch,
                                    lang="nl",
                                    speaker_embedding=True,
                                    cut_silences=False,
                                    return_language_id=True))

    datasets.append(TacotronDataset(build_path_to_transcript_dict_css10fr(),
                                    cache_dir=cache_dir_french,
                                    lang="fr",
                                    speaker_embedding=True,
                                    cut_silences=False,
                                    return_language_id=True))

    train_set = ConcatDataset(datasets)

    model = Tacotron2(spk_embed_dim=192,
                      language_embedding_amount=30,
                      initialize_from_pretrained_model=True,
                      initialize_multispeaker_projection=True)

    print("Training model")
    train_loop(net=model,
               train_dataset=train_set,
               device=device,
               save_directory=save_dir,
               steps=100000,
               batch_size=20,
               epochs_per_save=1,
               use_speaker_embedding=True,
               lang="en",
               lr=0.0005,
               path_to_checkpoint=resume_checkpoint,
               fine_tune=finetune,
               multi_ling=True)
