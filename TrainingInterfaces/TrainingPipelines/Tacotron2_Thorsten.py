import os
import random

import torch

from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.Tacotron2 import Tacotron2
from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.TacotronDataset import TacotronDataset
from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.tacotron2_train_loop import train_loop
from Utility.path_to_transcript_dicts import build_path_to_transcript_dict_thorsten as build_path_to_transcript_dict


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
    cache_dir = os.path.join("Corpora", "Thorsten")
    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join("Models", "Tacotron2_Thorsten")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path_to_transcript_dict = build_path_to_transcript_dict()

    train_set = TacotronDataset(path_to_transcript_dict,
                                cache_dir=cache_dir,
                                lang="de",
                                min_len_in_seconds=1,
                                max_len_in_seconds=10,
                                rebuild_cache=False,
                                cut_silences=True)

    model = Tacotron2(idim=166, odim=80, spk_embed_dim=None, use_dtw_loss=False)

    print("Training model")
    train_loop(net=model,
               train_dataset=train_set,
               device=device,
               save_directory=save_dir,
               steps=100000,
               batch_size=84,  # this works for a 24GB GPU. For a smaller GPU, consider decreasing batchsize.
               epochs_per_save=2,
               use_speaker_embedding=False,
               lang="de",
               lr=0.001,
               path_to_checkpoint=resume_checkpoint,
               fine_tune=finetune,
               resume=resume)
