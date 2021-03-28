"""
https://alexander-stasiuk.medium.com/pytorch-weights-averaging-e2c0fa611a0c
"""

import os

import torch

from MelGAN.MelGANGenerator import MelGANGenerator
from TransformerTTS.TransformerTTS import Transformer


def load_net_trans(path, idim=133, odim=80, spk_emb_dim=None):
    check_dict = torch.load(path, map_location=torch.device("cpu"))
    net = Transformer(idim=idim, odim=odim, spk_embed_dim=spk_emb_dim)
    net.load_state_dict(check_dict["model"])
    return net


def load_net_melgan(path):
    check_dict = torch.load(path, map_location=torch.device("cpu"))
    net = MelGANGenerator()
    net.load_state_dict(check_dict["generator"])
    return net


def get_n_recent_checkpoints_paths(checkpoint_dir, n=5):
    print("selecting checkpoints...")
    checkpoint_list = list()
    for el in os.listdir(checkpoint_dir):
        checkpoint_list.append(int(el.split(".")[0].split("_")[1]))
    checkpoint_list.sort(reverse=True)
    return [os.path.join(checkpoint_dir, "checkpoint_{}.pt".format(step)) for step in checkpoint_list[:n]]


def average_checkpoints(list_of_checkpoint_paths, load_func):
    snapshots_weights = {}
    model = None
    for path_to_checkpoint in list_of_checkpoint_paths:
        print("loading model {}".format(path_to_checkpoint))
        model = load_func(path=path_to_checkpoint)
        snapshots_weights[path_to_checkpoint] = dict(model.named_parameters())
    params = model.named_parameters()
    dict_params = dict(params)
    checkpoint_amount = len(snapshots_weights)
    print("averaging...")
    for name in dict_params.keys():
        custom_params = None
        for _, checkpoint_parameters in snapshots_weights.items():
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


def save_model_for_use(model, name="Transformer_English_Single.pt", dict_name="model"):
    print("saving model...")
    torch.save({dict_name: model.state_dict()}, name)
    print("...done!")


if __name__ == '__main__':
    checkpoint_paths = get_n_recent_checkpoints_paths(checkpoint_dir="../Models/Transformer_LJ", n=5)
    averaged_model = average_checkpoints(checkpoint_paths, load_func=load_net_trans)
    save_model_for_use(model=averaged_model, name="../Models/Use/Transformer_English_Single.pt")

    # checkpoint_paths = get_n_recent_checkpoints_paths(checkpoint_dir="../Models/Transformer_Hokus", n=5)
    # averaged_model = average_checkpoints(checkpoint_paths, load_func=load_net_trans)
    # save_model_for_use(model=averaged_model, name="../Models/Use/Transformer_German_Single.pt")

    checkpoint_paths = get_n_recent_checkpoints_paths(checkpoint_dir="../Models/melgan", n=3)
    averaged_model = average_checkpoints(checkpoint_paths, load_func=load_net_melgan)
    save_model_for_use(model=averaged_model, name="../Models/Use/MelGAN.pt", dict_name="generator")
