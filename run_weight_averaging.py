"""
https://alexander-stasiuk.medium.com/pytorch-weights-averaging-e2c0fa611a0c
"""

import os

import torch

from TrainingInterfaces.Spectrogram_to_Wave.HiFIGAN.HiFiGAN import HiFiGANGenerator
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2
from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.Tacotron2 import Tacotron2


def load_net_taco(path):
    check_dict = torch.load(path, map_location=torch.device("cpu"))
    net = Tacotron2()
    net.load_state_dict(check_dict["model"])
    return net


def load_net_fast(path):
    check_dict = torch.load(path, map_location=torch.device("cpu"))
    net = FastSpeech2()
    net.load_state_dict(check_dict["model"])
    return net


def load_net_hifigan(path):
    check_dict = torch.load(path, map_location=torch.device("cpu"))
    net = HiFiGANGenerator()
    net.load_state_dict(check_dict["generator"])
    return net


def get_n_recent_checkpoints_paths(checkpoint_dir, n=5):
    print("selecting checkpoints...")
    checkpoint_list = list()
    for el in os.listdir(checkpoint_dir):
        if el.endswith(".pt") and el != "best.pt":
            checkpoint_list.append(int(el.split(".")[0].split("_")[1]))
    if len(checkpoint_list) == 0:
        return None
    elif len(checkpoint_list) < n:
        n = len(checkpoint_list)
    checkpoint_list.sort(reverse=True)
    return [os.path.join(checkpoint_dir, "checkpoint_{}.pt".format(step)) for step in checkpoint_list[:n]]


def average_checkpoints(list_of_checkpoint_paths, load_func):
    if list_of_checkpoint_paths is None:
        return None
    checkpoints_weights = {}
    model = None
    for path_to_checkpoint in list_of_checkpoint_paths:
        print("loading model {}".format(path_to_checkpoint))
        model = load_func(path=path_to_checkpoint)
        checkpoints_weights[path_to_checkpoint] = dict(model.named_parameters())
    params = model.named_parameters()
    dict_params = dict(params)
    checkpoint_amount = len(checkpoints_weights)
    print("averaging...")
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


def save_model_for_use(model, name="", dict_name="model"):
    if model is None:
        return
    print("saving model...")
    torch.save({dict_name: model.state_dict()}, name)
    print("...done!")


def make_best_in_all(n=3):
    for model_dir in os.listdir("Models"):
        if "HiFiGAN" in model_dir:
            checkpoint_paths = get_n_recent_checkpoints_paths(checkpoint_dir="Models/{}".format(model_dir), n=n)
            averaged_model = average_checkpoints(checkpoint_paths, load_func=load_net_hifigan)
            save_model_for_use(model=averaged_model, name="Models/{}/best.pt".format(model_dir), dict_name="generator")
        elif "Tacotron2" in model_dir:
            checkpoint_paths = get_n_recent_checkpoints_paths(checkpoint_dir="Models/{}".format(model_dir), n=n)
            averaged_model = average_checkpoints(checkpoint_paths, load_func=load_net_taco)
            save_model_for_use(model=averaged_model, name="Models/{}/best.pt".format(model_dir))
        elif "FastSpeech2" in model_dir:
            checkpoint_paths = get_n_recent_checkpoints_paths(checkpoint_dir="Models/{}".format(model_dir), n=n)
            averaged_model = average_checkpoints(checkpoint_paths, load_func=load_net_fast)
            save_model_for_use(model=averaged_model, name="Models/{}/best.pt".format(model_dir))


def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def show_all_models_params():
    from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.Tacotron2 import Tacotron2
    from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2
    print("Number of (trainable) Parameters in Tacotron2: {}".format(count_parameters(Tacotron2())))
    print("Number of (trainable) Parameters in FastSpeech2: {}".format(count_parameters(FastSpeech2())))


if __name__ == '__main__':
    # show_all_models_params()
    make_best_in_all(n=5)
