"""
https://alexander-stasiuk.medium.com/pytorch-weights-averaging-e2c0fa611a0c
"""

import os

import torch

from TrainingInterfaces.Spectrogram_to_Wave.BigVGAN.BigVGAN import BigVGAN
from TrainingInterfaces.Spectrogram_to_Wave.HiFiGAN.HiFiGAN import HiFiGANGenerator
from TrainingInterfaces.Text_to_Spectrogram.StochasticToucanTTS.StochasticToucanTTS import StochasticToucanTTS
from TrainingInterfaces.Text_to_Spectrogram.ToucanTTS.ToucanTTS import ToucanTTS
from Utility.storage_config import MODELS_DIR


def load_net_toucan(path):
    check_dict = torch.load(path, map_location=torch.device("cpu"))
    try:
        try:
            net = ToucanTTS()
            net.load_state_dict(check_dict["model"])
        except RuntimeError:
            try:
                net = ToucanTTS(lang_embs=None)
                net.load_state_dict(check_dict["model"])
            except RuntimeError:
                net = ToucanTTS(lang_embs=None, utt_embed_dim=None)
                net.load_state_dict(check_dict["model"])
    except RuntimeError:
        try:
            net = StochasticToucanTTS()
            net.load_state_dict(check_dict["model"])
        except RuntimeError:
            try:
                net = StochasticToucanTTS(lang_embs=None)
                net.load_state_dict(check_dict["model"])
            except RuntimeError:
                net = StochasticToucanTTS(lang_embs=None, utt_embed_dim=None)
                net.load_state_dict(check_dict["model"])
    return net, check_dict["default_emb"]


def load_net_hifigan(path):
    check_dict = torch.load(path, map_location=torch.device("cpu"))
    net = HiFiGANGenerator()
    net.load_state_dict(check_dict["generator"])
    return net, None  # does not have utterance embedding


def load_net_bigvgan(path):
    check_dict = torch.load(path, map_location=torch.device("cpu"))
    net = BigVGAN()
    net.load_state_dict(check_dict["generator"])
    return net, None  # does not have utterance embedding


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
    if default_embed is None:
        # HiFiGAN case
        torch.save({dict_name: model.state_dict()}, name)
    else:
        # TTS case
        torch.save({dict_name: model.state_dict(), "default_emb": default_embed}, name)
    print("...done!")


def make_best_in_all():
    for model_dir in os.listdir(MODELS_DIR):
        if os.path.isdir(os.path.join(MODELS_DIR, model_dir)):
            if "HiFiGAN" in model_dir or "Avocodo" in model_dir:
                checkpoint_paths = get_n_recent_checkpoints_paths(checkpoint_dir=os.path.join(MODELS_DIR, model_dir), n=3)
                if checkpoint_paths is None:
                    continue
                averaged_model, _ = average_checkpoints(checkpoint_paths, load_func=load_net_hifigan)
                save_model_for_use(model=averaged_model, name=os.path.join(MODELS_DIR, model_dir, "best.pt"), dict_name="generator")

            elif "BigVGAN" in model_dir:
                checkpoint_paths = get_n_recent_checkpoints_paths(checkpoint_dir=os.path.join(MODELS_DIR, model_dir), n=3)
                if checkpoint_paths is None:
                    continue
                averaged_model, _ = average_checkpoints(checkpoint_paths, load_func=load_net_bigvgan)
                save_model_for_use(model=averaged_model, name=os.path.join(MODELS_DIR, model_dir, "best.pt"), dict_name="generator")
            elif "ToucanTTS" in model_dir:
                checkpoint_paths = get_n_recent_checkpoints_paths(checkpoint_dir=os.path.join(MODELS_DIR, model_dir), n=3)
                if checkpoint_paths is None:
                    continue
                averaged_model, default_embed = average_checkpoints(checkpoint_paths, load_func=load_net_toucan)
                save_model_for_use(model=averaged_model, default_embed=default_embed, name=os.path.join(MODELS_DIR, model_dir, "best.pt"))


def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def show_all_models_params():
    from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2
    print("Number of (trainable) Parameters in FastSpeech2: {}".format(count_parameters(FastSpeech2())))
    from TrainingInterfaces.Spectrogram_to_Embedding.StyleEmbedding import StyleEmbedding
    print("Number of (trainable) Parameters in GST: {}".format(count_parameters(StyleEmbedding())))
    from TrainingInterfaces.Spectrogram_to_Wave.HiFiGAN.HiFiGAN import HiFiGANGenerator
    print("Number of (trainable) Parameters in the HiFiGAN Generator: {}".format(count_parameters(HiFiGANGenerator())))
    from TrainingInterfaces.Spectrogram_to_Wave.BigVGAN.BigVGAN import BigVGAN
    print("Number of (trainable) Parameters in the BigVGAN Generator: {}".format(count_parameters(BigVGAN())))


if __name__ == '__main__':
    make_best_in_all()
