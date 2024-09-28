"""
https://alexander-stasiuk.medium.com/pytorch-weights-averaging-e2c0fa611a0c
"""

import os

import torch

from Modules.ToucanTTS.InferenceToucanTTS import ToucanTTS
from Modules.Vocoder.HiFiGAN_Generator import HiFiGAN


def load_net_toucan(path):
    check_dict = torch.load(path, map_location=torch.device("cpu"))
    net = ToucanTTS(weights=check_dict["model"], config=check_dict["config"])
    return net, check_dict["default_emb"]


def load_net_bigvgan(path):
    check_dict = torch.load(path, map_location=torch.device("cpu"))
    net = HiFiGAN(weights=check_dict["generator"])
    return net, None


def get_n_recent_checkpoints_paths(checkpoint_dir, n=5):
    print("selecting checkpoints...")
    checkpoint_list = list()
    for el in os.listdir(checkpoint_dir):
        if el.endswith(".pt") and el.startswith("checkpoint_"):
            try:
                checkpoint_list.append(int(el.split(".")[0].split("_")[1]))
            except RuntimeError:
                pass
    if len(checkpoint_list) == 0:
        return None
    elif len(checkpoint_list) < n:
        n = len(checkpoint_list)
    checkpoint_list.sort(reverse=True)
    return [os.path.join(checkpoint_dir, "checkpoint_{}.pt".format(step)) for step in checkpoint_list[:n]]


def average_checkpoints(list_of_checkpoint_paths, load_func):
    # COLLECT CHECKPOINTS
    if list_of_checkpoint_paths is None or len(list_of_checkpoint_paths) == 0:
        return None
    checkpoints_weights = {}
    model = None
    default_embed = None

    # LOAD CHECKPOINTS
    for path_to_checkpoint in list_of_checkpoint_paths:
        print("loading model {}".format(path_to_checkpoint))
        model, default_embed = load_func(path=path_to_checkpoint)
        checkpoints_weights[path_to_checkpoint] = dict(model.named_parameters())

    # AVERAGE CHECKPOINTS
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
    return model, default_embed


def save_model_for_use(model, name="", default_embed=None, dict_name="model"):
    print("saving model...")
    torch.save({dict_name: model.state_dict(), "default_emb": default_embed, "config": model.config}, name)
    print("...done!")

def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

