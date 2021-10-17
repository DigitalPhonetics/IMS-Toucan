import random

import torch
import torch.multiprocessing

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

    print("Preparing")
    cache_dir_english_nancy = os.path.join("Corpora", "meta_English_nancy")
    model_save_dirs.append(os.path.join("Models", "Meta_Waldweihe", "meta_English_nancy"))
    os.makedirs(cache_dir_english_nancy, exist_ok=True)
    languages.append("en")

    cache_dir_english_lj = os.path.join("Corpora", "meta_English_lj")
    model_save_dirs.append(os.path.join("Models", "Meta_Waldweihe", "meta_English_lj"))
    os.makedirs(cache_dir_english_lj, exist_ok=True)
    languages.append("en")

    cache_dir_greek = os.path.join("Corpora", "meta_Greek")
    model_save_dirs.append(os.path.join("Models", "Meta_Waldweihe", "meta_Greek"))
    os.makedirs(cache_dir_greek, exist_ok=True)
    languages.append("el")

    cache_dir_spanish = os.path.join("Corpora", "meta_Spanish")
    model_save_dirs.append(os.path.join("Models", "Meta_Waldweihe", "meta_Spanish"))
    os.makedirs(cache_dir_spanish, exist_ok=True)
    languages.append("es")

    cache_dir_finnish = os.path.join("Corpora", "meta_Finnish")
    model_save_dirs.append(os.path.join("Models", "Meta_Waldweihe", "meta_Finnish"))
    os.makedirs(cache_dir_finnish, exist_ok=True)
    languages.append("fi")

    cache_dir_russian = os.path.join("Corpora", "meta_Russian")
    model_save_dirs.append(os.path.join("Models", "Meta_Waldweihe", "meta_Russian"))
    os.makedirs(cache_dir_russian, exist_ok=True)
    languages.append("ru")

    cache_dir_hungarian = os.path.join("Corpora", "meta_Hungarian")
    model_save_dirs.append(os.path.join("Models", "Meta_Waldweihe", "meta_Hungarian"))
    os.makedirs(cache_dir_hungarian, exist_ok=True)
    languages.append("hu")

    cache_dir_dutch = os.path.join("Corpora", "meta_Dutch")
    model_save_dirs.append(os.path.join("Models", "Meta_Waldweihe", "meta_Dutch"))
    os.makedirs(cache_dir_dutch, exist_ok=True)
    languages.append("nl")

    cache_dir_french = os.path.join("Corpora", "meta_French")
    model_save_dirs.append(os.path.join("Models", "Meta_Waldweihe", "meta_French"))
    os.makedirs(cache_dir_french, exist_ok=True)
    languages.append("fr")

    meta_save_dir = os.path.join("Models", "Meta_Waldweihe", "Tacotron2_MetaCheckpoint")
    os.makedirs(meta_save_dir, exist_ok=True)

    datasets = list()

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
    gpu = ["7"]
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu)

    index = 5

    instance_save_dir = model_save_dirs[index] + f"_iteration_{0}"
    os.makedirs(instance_save_dir, exist_ok=True)
    batchsize = 24
    batches_per_epoch = max((len(datasets[index]) // batchsize), 1)  # max with one to avoid zero division
    epochs_per_save = max(round(100 / batches_per_epoch), 1)  # just to balance the amount of checkpoints
    train_loop(net=Tacotron2(use_alignment_loss=False),
               train_dataset=datasets[index],
               device=torch.device(f"cuda:{gpu}"),
               save_directory=instance_save_dir,
               steps=3000,
               batch_size=batchsize,
               epochs_per_save=epochs_per_save,
               lang=languages[index],
               lr=0.001,
               path_to_checkpoint=meta_save_dir + "/best.pt",
               fine_tune=not resume,
               resume=resume,
               cycle_loss_start_steps=None  # not used here, only for final adaptation
               )
    # meta_model = average_models(individual_models)
    # torch.save({'model': meta_model.state_dict()}, meta_save_dir + "/best.pt")


def average_models(models):
    checkpoints_weights = {}
    model = None
    for index, model in enumerate(models):
        checkpoints_weights[index] = dict(model.named_parameters())
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
