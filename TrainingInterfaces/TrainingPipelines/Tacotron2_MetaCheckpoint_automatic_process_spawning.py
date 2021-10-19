import random

import torch
import torch.multiprocessing
from torch import multiprocessing as mp

from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.Tacotron2 import Tacotron2
from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.TacotronDataset import TacotronDataset
from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.tacotron2_train_loop import train_loop
from Utility.path_to_transcript_dicts import *


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume):
    # =================
    verbose = False
    # =================

    torch.manual_seed(131714)
    random.seed(131714)
    torch.random.manual_seed(131714)

    model_save_dirs = list()
    languages = list()
    datasets = list()

    base_dir = os.path.join("Models", "Meta_multi")

    print("Preparing")
    cache_dir_english_nancy = os.path.join("Corpora", "meta_English_nancy")
    model_save_dirs.append(os.path.join(base_dir, "meta_English_nancy"))
    os.makedirs(cache_dir_english_nancy, exist_ok=True)
    languages.append("en")

    cache_dir_english_lj = os.path.join("Corpora", "meta_English_lj")
    model_save_dirs.append(os.path.join(base_dir, "meta_English_lj"))
    os.makedirs(cache_dir_english_lj, exist_ok=True)
    languages.append("en")

    cache_dir_greek = os.path.join("Corpora", "meta_Greek")
    model_save_dirs.append(os.path.join(base_dir, "meta_Greek"))
    os.makedirs(cache_dir_greek, exist_ok=True)
    languages.append("el")

    cache_dir_spanish = os.path.join("Corpora", "meta_Spanish")
    model_save_dirs.append(os.path.join(base_dir, "meta_Spanish"))
    os.makedirs(cache_dir_spanish, exist_ok=True)
    languages.append("es")

    cache_dir_finnish = os.path.join("Corpora", "meta_Finnish")
    model_save_dirs.append(os.path.join(base_dir, "meta_Finnish"))
    os.makedirs(cache_dir_finnish, exist_ok=True)
    languages.append("fi")

    cache_dir_russian = os.path.join("Corpora", "meta_Russian")
    model_save_dirs.append(os.path.join(base_dir, "meta_Russian"))
    os.makedirs(cache_dir_russian, exist_ok=True)
    languages.append("ru")

    cache_dir_hungarian = os.path.join("Corpora", "meta_Hungarian")
    model_save_dirs.append(os.path.join(base_dir, "meta_Hungarian"))
    os.makedirs(cache_dir_hungarian, exist_ok=True)
    languages.append("hu")

    cache_dir_dutch = os.path.join("Corpora", "meta_Dutch")
    model_save_dirs.append(os.path.join(base_dir, "meta_Dutch"))
    os.makedirs(cache_dir_dutch, exist_ok=True)
    languages.append("nl")

    cache_dir_french = os.path.join("Corpora", "meta_French")
    model_save_dirs.append(os.path.join(base_dir, "meta_French"))
    os.makedirs(cache_dir_french, exist_ok=True)
    languages.append("fr")

    meta_save_dir = os.path.join(base_dir, "Tacotron2_MetaCheckpoint")
    os.makedirs(meta_save_dir, exist_ok=True)

    datasets.append(TacotronDataset(build_path_to_transcript_dict_nancy(),
                                    cache_dir=cache_dir_english_nancy,
                                    lang="en",
                                    loading_processes=20,  # run this on a lonely server at night
                                    cut_silences=True,
                                    min_len_in_seconds=2,  # needs to be long enough for the speaker embedding in the cycle objective to make sense
                                    max_len_in_seconds=13))

    datasets.append(TacotronDataset(build_path_to_transcript_dict_ljspeech(),
                                    cache_dir=cache_dir_english_lj,
                                    lang="en",
                                    loading_processes=20,
                                    cut_silences=True,
                                    min_len_in_seconds=2,
                                    max_len_in_seconds=13))

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
    gpus_usable = ["4", "5", "6", "7", "8", "5", "6", "7", "8"]
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(",".join(gpus_usable))
    gpus_available = list(range(len(gpus_usable)))
    gpus_in_use = []

    for iteration in range(10):

        processes = list()
        individual_models = list()
        torch.cuda.empty_cache()

        if iteration == 0:
            # make sure all models train with the same initialization
            torch.save({'model': Tacotron2(use_alignment_loss=False).state_dict()}, meta_save_dir + f"/meta_{iteration}it.pt")

        for index, train_set in enumerate(datasets):
            instance_save_dir = model_save_dirs[index] + f"_iteration_{iteration}"
            os.makedirs(instance_save_dir, exist_ok=True)
            batchsize = 24
            batches_per_epoch = max((len(train_set) // batchsize), 1)  # max with one to avoid zero division
            epochs_per_save = max(round(100 / batches_per_epoch), 1)  # just to balance the amount of checkpoints
            individual_models.append(Tacotron2(use_alignment_loss=False))
            processes.append(mp.Process(target=train_loop,
                                        kwargs={
                                            "net"                   : individual_models[-1],
                                            "train_dataset"         : train_set,
                                            "device"                : torch.device(f"cuda:{gpus_available[-1]}"),
                                            "save_directory"        : instance_save_dir,
                                            "steps"                 : 6000,
                                            "batch_size"            : batchsize,
                                            "epochs_per_save"       : epochs_per_save,
                                            "lang"                  : languages[index],
                                            "lr"                    : 0.001,
                                            "path_to_checkpoint"    : meta_save_dir + f"/meta_{iteration}it.pt",
                                            "fine_tune"             : not resume,
                                            "resume"                : resume,
                                            "cycle_loss_start_steps": None,  # not used here, only for final adaptation
                                            "silent"                : True
                                            }))
            processes[-1].start()
            if verbose:
                print(f"Starting {instance_save_dir} on cuda:{gpus_available[-1]}")
            gpus_in_use.append(gpus_available.pop())
            while len(gpus_available) == 0:
                if verbose:
                    print("All GPUs available should be filled now. Waiting for one process to finish to start the next one.")
                processes[0].join()
                processes.pop(0)
                gpus_available.append(gpus_in_use.pop(0))

        if verbose:
            print("Waiting for the remainders to finish...")
        for process in processes:
            process.join()
            gpus_available.append(gpus_in_use.pop(0))

        meta_model = average_models(individual_models)
        torch.save({'model': meta_model.state_dict()}, meta_save_dir + f"/meta_{iteration + 1}it.pt")


def average_models(models):
    checkpoints_weights = {}
    model = None
    for index, model in enumerate(models):
        checkpoints_weights[index] = dict(model.named_parameters())
    model = model.cpu()
    params = model.named_parameters()
    dict_params = dict(params)
    checkpoint_amount = len(checkpoints_weights)
    print("\n\naveraging...\n\n")
    for name in dict_params.keys():
        custom_params = None
        for _, checkpoint_parameters in checkpoints_weights.items():
            if custom_params is None:
                custom_params = checkpoint_parameters[name].data
            else:
                custom_params += checkpoint_parameters[name].data
        dict_params[name].data.copy_(custom_params / checkpoint_amount)
    model_dict = model.state_dict()
    model_dict.update(dict_params)
    model.load_state_dict(model_dict)
    model.eval()
    return model
