import random

import torch
from torch.utils.data import ConcatDataset

from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeechDataset import FastSpeechDataset
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.fastspeech2_train_loop import train_loop
from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.Tacotron2 import Tacotron2
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
    cache_dir_hifitts = os.path.join("Corpora", "multispeaker_nvidia_hifitts")
    os.makedirs(cache_dir_hifitts, exist_ok=True)

    cache_dir_nancy = os.path.join("Corpora", "multispeaker_nancy")
    os.makedirs(cache_dir_nancy, exist_ok=True)

    cache_dir_lj = os.path.join("Corpora", "multispeaker_lj")
    os.makedirs(cache_dir_lj, exist_ok=True)

    cache_dir_libri = os.path.join("Corpora", "multispeaker_libri")
    os.makedirs(cache_dir_libri, exist_ok=True)

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join("Models", "FastSpeech2_MultiEnglish")
    os.makedirs(save_dir, exist_ok=True)

    acoustic_model = Tacotron2(spk_embed_dim=192)
    acoustic_model.load_state_dict(torch.load(os.path.join("Models", "Tacotron2_MultiEnglish", "best.pt"), map_location='cpu')["model"])

    datasets = list()

    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_nvidia_hifitts(),
                                      cache_dir=cache_dir_hifitts,
                                      lang="en",
                                      speaker_embedding=True,
                                      return_language_id=False,
                                      device=device,
                                      acoustic_model=acoustic_model))

    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_nancy(),
                                      cache_dir=cache_dir_nancy,
                                      lang="en",
                                      speaker_embedding=True,
                                      return_language_id=False,
                                      device=device,
                                      acoustic_model=acoustic_model))

    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_ljspeech(),
                                      cache_dir=cache_dir_lj,
                                      lang="en",
                                      speaker_embedding=True,
                                      return_language_id=False,
                                      device=device,
                                      acoustic_model=acoustic_model))

    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_libritts(),
                                      cache_dir=cache_dir_libri,
                                      lang="en",
                                      speaker_embedding=True,
                                      return_language_id=False,
                                      device=device,
                                      acoustic_model=acoustic_model))

    train_set = ConcatDataset(datasets)
    del acoustic_model

    model = FastSpeech2(spk_embed_dim=192, initialize_from_pretrained_encoder=True)

    print("Training model")
    train_loop(net=model,
               train_dataset=train_set,
               device=device,
               save_directory=save_dir,
               steps=50000,
               batch_size=20,
               use_speaker_embedding=True,
               lang="en",
               lr=0.0001,
               warmup_steps=14000,
               path_to_checkpoint=resume_checkpoint,
               fine_tune=finetune,
               freeze_encoder_until=22000)
