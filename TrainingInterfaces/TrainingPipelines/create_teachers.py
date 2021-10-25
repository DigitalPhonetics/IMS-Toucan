import random

import torch
import torch.multiprocessing
from torch import multiprocessing as mp

from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.Tacotron2 import Tacotron2
from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.TacotronDataset import TacotronDataset
from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.tacotron2_train_loop import train_loop
from Utility.path_to_transcript_dicts import *


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume):
    torch.manual_seed(131714)
    random.seed(131714)
    torch.random.manual_seed(131714)

    model_save_dirs = list()
    languages = list()
    datasets = list()

    print("Preparing")

    cache_dir_greek = os.path.join("Corpora", "meta_Greek")
    model_save_dirs.append(os.path.join("Models", "Tacotron2_Greek_Aligner"))
    os.makedirs(cache_dir_greek, exist_ok=True)
    languages.append("el")

    cache_dir_spanish = os.path.join("Corpora", "meta_Spanish")
    model_save_dirs.append(os.path.join("Models", "Tacotron2_Spanish_Aligner"))
    os.makedirs(cache_dir_spanish, exist_ok=True)
    languages.append("es")

    cache_dir_finnish = os.path.join("Corpora", "meta_Finnish")
    model_save_dirs.append(os.path.join("Models", "Tacotron2_Finnish_Aligner"))
    os.makedirs(cache_dir_finnish, exist_ok=True)
    languages.append("fi")

    cache_dir_russian = os.path.join("Corpora", "meta_Russian")
    model_save_dirs.append(os.path.join("Models", "Tacotron2_Russian_Aligner"))
    os.makedirs(cache_dir_russian, exist_ok=True)
    languages.append("ru")

    cache_dir_hungarian = os.path.join("Corpora", "meta_Hungarian")
    model_save_dirs.append(os.path.join("Models", "Tacotron2_Hungarian_Aligner"))
    os.makedirs(cache_dir_hungarian, exist_ok=True)
    languages.append("hu")

    cache_dir_dutch = os.path.join("Corpora", "meta_Dutch")
    model_save_dirs.append(os.path.join("Models", "Tacotron2_Dutch_Aligner"))
    os.makedirs(cache_dir_dutch, exist_ok=True)
    languages.append("nl")

    cache_dir_french = os.path.join("Corpora", "meta_French")
    model_save_dirs.append(os.path.join("Models", "Tacotron2_French_Aligner"))
    os.makedirs(cache_dir_french, exist_ok=True)
    languages.append("fr")


    datasets.append(TacotronDataset(build_path_to_transcript_dict_css10el(),
                                    cache_dir=cache_dir_greek,
                                    lang="el",
                                    loading_processes=20,
                                    cut_silences=True,
                                    min_len_in_seconds=2,
                                    max_len_in_seconds=13))

    datasets.append(TacotronDataset(build_path_to_transcript_dict_css10es(),
                                    cache_dir=cache_dir_spanish,
                                    lang="es",
                                    loading_processes=20,
                                    cut_silences=True,
                                    min_len_in_seconds=2,
                                    max_len_in_seconds=13))

    datasets.append(TacotronDataset(build_path_to_transcript_dict_css10fi(),
                                    cache_dir=cache_dir_finnish,
                                    lang="fi",
                                    loading_processes=20,
                                    cut_silences=True,
                                    min_len_in_seconds=2,
                                    max_len_in_seconds=13))

    datasets.append(TacotronDataset(build_path_to_transcript_dict_css10ru(),
                                    cache_dir=cache_dir_russian,
                                    lang="ru",
                                    loading_processes=20,
                                    cut_silences=True,
                                    min_len_in_seconds=2,
                                    max_len_in_seconds=13))

    datasets.append(TacotronDataset(build_path_to_transcript_dict_css10hu(),
                                    cache_dir=cache_dir_hungarian,
                                    lang="hu",
                                    loading_processes=20,
                                    cut_silences=True,
                                    min_len_in_seconds=2,
                                    max_len_in_seconds=13))

    datasets.append(TacotronDataset(build_path_to_transcript_dict_css10nl(),
                                    cache_dir=cache_dir_dutch,
                                    lang="nl",
                                    loading_processes=20,
                                    cut_silences=True,
                                    min_len_in_seconds=2,
                                    max_len_in_seconds=13))

    datasets.append(TacotronDataset(build_path_to_transcript_dict_css10fr(),
                                    cache_dir=cache_dir_french,
                                    lang="fr",
                                    loading_processes=20,
                                    cut_silences=True,
                                    min_len_in_seconds=2,
                                    max_len_in_seconds=13))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    gpus_usable = ["4,5,6,7"]
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(",".join(gpus_usable))
    gpus_available = [0,0,1,1,2,2,3,3]
    gpus_in_use = []

    processes = list()
    for index, train_set in enumerate(datasets):
        instance_save_dir = model_save_dirs[index]
        os.makedirs(instance_save_dir, exist_ok=True)
        batchsize = 24
        batches_per_epoch = max((len(train_set) // batchsize), 1)  # max with one to avoid zero division
        epochs_per_save = max(round(100 / batches_per_epoch), 1)  # just to balance the amount of checkpoints
        processes.append(mp.Process(target=train_loop,
                                    kwargs={
                                        "net": Tacotron2(elayers=0, econv_layers=0, adim=256, embed_dim=256, prenet_layers=0, postnet_layers=0),
                                        "train_dataset": train_set,
                                        "device": torch.device(f"cuda:{gpus_available[-1]}"),
                                        "save_directory": instance_save_dir,
                                        "steps": 20000,
                                        "batch_size": batchsize,
                                        "epochs_per_save": epochs_per_save,
                                        "lang": languages[index],
                                        "lr": 0.001,
                                        "path_to_checkpoint": "Models/Tacotron2_Nancy_Aligner/best.pt",
                                        "fine_tune": True,
                                        "resume": resume,
                                        "cycle_loss_start_steps": None,  # not used here, only for final adaptation
                                        "silent": True
                                    }))
        processes[-1].start()
        print(f"Starting {instance_save_dir} on cuda:{gpus_available[-1]}")
        gpus_in_use.append(gpus_available.pop())
        while len(gpus_available) == 0:
            print("All GPUs available should be filled now. Waiting for one process to finish to start the next one.")
            processes[0].join()
            processes.pop(0)
            gpus_available.append(gpus_in_use.pop(0))

    print("Waiting for the remainders to finish...")
    for process in processes:
        process.join()
        gpus_available.append(gpus_in_use.pop(0))
