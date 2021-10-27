import random

import torch
import torch.multiprocessing

from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeechDataset import FastSpeechDataset
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.fastspeech2_train_loop import train_loop
from Utility.path_to_transcript_dicts import *


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume):
    # =================
    verbose = False
    # =================

    torch.manual_seed(131714)
    random.seed(131714)
    torch.random.manual_seed(131714)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"  # fastspeech is fast enough that we don't have to multiprocess everything

    model_save_dirs = list()
    languages = list()
    datasets = list()

    base_dir = os.path.join("Models", "First_Order_LAML_FastSpeech2")

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

    meta_save_dir = os.path.join(base_dir, "FastSpeech2_MetaCheckpoint")
    os.makedirs(meta_save_dir, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"

    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_nancy(),
                                      acoustic_checkpoint_path="Models/Tacotron2_Nancy_Aligner/best.pt",
                                      cache_dir=cache_dir_english_nancy,
                                      device=torch.device("cuda"),
                                      lang="en"))

    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_css10el(),
                                      acoustic_checkpoint_path="Models/Tacotron2_Greek_Aligner/best.pt",
                                      cache_dir=cache_dir_greek,
                                      device=torch.device("cuda"),
                                      lang="el"))

    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_css10es(),
                                      acoustic_checkpoint_path="Models/Tacotron2_Spanish_Aligner/best.pt",
                                      cache_dir=cache_dir_spanish,
                                      device=torch.device("cuda"),
                                      lang="es"))

    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_css10fi(),
                                      acoustic_checkpoint_path="Models/Tacotron2_Finnish_Aligner/best.pt",
                                      cache_dir=cache_dir_finnish,
                                      device=torch.device("cuda"),
                                      lang="fi"))

    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_css10ru(),
                                      acoustic_checkpoint_path="Models/Tacotron2_Russian_Aligner/best.pt",
                                      cache_dir=cache_dir_russian,
                                      device=torch.device("cuda"),
                                      lang="ru"))

    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_css10hu(),
                                      acoustic_checkpoint_path="Models/Tacotron2_Hungarian_Aligner/best.pt",
                                      cache_dir=cache_dir_hungarian,
                                      device=torch.device("cuda"),
                                      lang="hu"))

    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_css10nl(),
                                      acoustic_checkpoint_path="Models/Tacotron2_Dutch_Aligner/best.pt",
                                      cache_dir=cache_dir_dutch,
                                      device=torch.device("cuda"),
                                      lang="nl"))

    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_css10fr(),
                                      acoustic_checkpoint_path="Models/Tacotron2_French_Aligner/best.pt",
                                      cache_dir=cache_dir_french,
                                      device=torch.device("cuda"),
                                      lang="fr"))

    for iteration in range(100):

        individual_models = list()

        if iteration == 0:
            # make sure all models train with the same initialization
            torch.save({'model': FastSpeech2().state_dict()}, meta_save_dir + f"/meta_{iteration}it.pt")

        for index, train_set in enumerate(datasets):
            instance_save_dir = model_save_dirs[index] + f"_iteration_{iteration}"
            os.makedirs(instance_save_dir, exist_ok=True)
            batchsize = 32
            batches_per_epoch = max((len(train_set) // batchsize), 1)  # max with one to avoid zero division
            epochs_per_save = max(round(500 / batches_per_epoch), 1)  # just to balance the amount of checkpoints
            individual_models.append(FastSpeech2())
            train_loop(net=individual_models[-1],
                       train_dataset=train_set,
                       device=torch.device("cuda"),
                       save_directory=instance_save_dir,
                       steps=iteration * 500 + 500,  # to make the latent spaces stay closer together initially
                       batch_size=batchsize,
                       epochs_per_save=epochs_per_save,
                       lang=languages[index],
                       lr=0.001,
                       path_to_checkpoint=meta_save_dir + f"/meta_{iteration}it.pt",
                       fine_tune=not resume,
                       resume=resume,
                       cycle_loss_start_steps=None  # not used here, only for final adaptation
                       )
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
