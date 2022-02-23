import random

import soundfile
import torch

from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.low_resource_fastspeech2_train_loop import train_loop
from Utility.corpus_preparation import prepare_fastspeech_corpus
from Utility.path_to_transcript_dicts import *


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume):
    if gpu_id == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")

    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
        device = torch.device("cuda")

    torch.manual_seed(131714)
    random.seed(131714)
    torch.random.manual_seed(131714)

    print("Preparing")

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join("Models", "FastSpeech2_Russian_low_resource")
    os.makedirs(save_dir, exist_ok=True)

    path_to_transcript_dict_ = build_path_to_transcript_dict_css10ru()
    path_to_transcript_dict = dict()

    paths = list(path_to_transcript_dict_.keys())
    used_samples = set()
    total_len = 0.0
    while total_len < 5.0 * 60.0:
        path = random.choice(paths)
        x, sr = soundfile.read(path)
        duration = len(x) / sr
        if 10 > duration > 5 and path not in used_samples:
            used_samples.add(path)
            total_len += duration

    print(f"Collected {total_len / 60.0} minutes worth of samples.")

    for key in path_to_transcript_dict_:
        if key in used_samples:
            path_to_transcript_dict[key] = path_to_transcript_dict_[key]

    train_set = prepare_fastspeech_corpus(transcript_dict=path_to_transcript_dict,
                                          corpus_dir=os.path.join("Corpora", "Russian_low_resource"),
                                          lang="ru")

    model = FastSpeech2(lang_embs=100)
    # because we want to finetune it, we treat it as multilingual and multispeaker model, even though it only has one speaker

    print("Training model")
    train_loop(net=model,
               train_dataset=train_set,
               device=device,
               save_directory=save_dir,
               steps=10000,
               batch_size=32,
               lang="ru",
               lr=0.00005,
               epochs_per_save=20,
               path_to_checkpoint="Models/FastSpeech2_Meta_no_slav/best.pt",
               fine_tune=True,
               resume=resume)
