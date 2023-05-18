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
                try:
                    net = ToucanTTS(lang_embs=None, utt_embed_dim=None)
                    net.load_state_dict(check_dict["model"])
                except RuntimeError:
                    try:
                        net = ToucanTTS(lang_embs=None, utt_embed_dim=512)
                        net.load_state_dict(check_dict["model"]) # xvect
                    except RuntimeError:
                        try:
                            print("Loading word emb architecture")
                            net = ToucanTTS(word_embed_dim=768)
                            net.load_state_dict(check_dict["model"])
                        except RuntimeError:
                            try:
                                print("Loading word emb architecture")
                                net = ToucanTTS(word_embed_dim=768, utt_embed_dim=None, lang_embs=None)
                                net.load_state_dict(check_dict["model"])
                            except RuntimeError:
                                try:
                                    print("Loading sent word emb architecture")
                                    net = ToucanTTS(sent_embed_dim=768,
                                                    sent_embed_adaptation="noadapt" not in path,
                                                    sent_embed_encoder=True,
                                                    use_sent_style_loss="loss" in path,
                                                    pre_embed="_pre" in path,
                                                    style_sent=True,
                                                    word_embed_dim=768)
                                    net.load_state_dict(check_dict["model"])
                                except RuntimeError:
                                    print("Loading sent emb architecture")
                                    lang_embs=None
                                    utt_embed_dim=512 if "_xvect" in path else 64

                                    if "laser" in path:
                                        sent_embed_dim = 1024
                                    if "lealla" in path:
                                        sent_embed_dim = 192
                                    if "para" in path:
                                        sent_embed_dim = 768
                                    if "mpnet" in path:
                                        sent_embed_dim = 768
                                    if "bertcls" in path:
                                        sent_embed_dim = 768
                                    if "bertlm" in path:
                                        sent_embed_dim = 768
                                    if "emoBERTcls" in path:
                                        sent_embed_dim = 768
                                    
                                    sent_embed_encoder=False
                                    sent_embed_decoder=False
                                    sent_embed_each=False
                                    sent_embed_postnet=False
                                    concat_sent_style=False
                                    use_concat_projection=False
                                    style_sent=False
                                    if "a01" in path:
                                        sent_embed_encoder=True
                                    if "a02" in path:
                                        sent_embed_encoder=True
                                        sent_embed_decoder=True
                                    if "a03" in path:
                                        sent_embed_encoder=True
                                        sent_embed_decoder=True
                                        sent_embed_postnet=True
                                    if "a04" in path:
                                        sent_embed_encoder=True
                                        sent_embed_each=True
                                    if "a05" in path:
                                        sent_embed_encoder=True
                                        sent_embed_decoder=True
                                        sent_embed_each=True
                                    if "a06" in path:
                                        sent_embed_encoder=True
                                        sent_embed_decoder=True
                                        sent_embed_each=True
                                        sent_embed_postnet=True
                                    if "a07" in path:
                                        concat_sent_style=True
                                        use_concat_projection=True
                                    if "a08" in path:
                                        concat_sent_style=True
                                    if "a09" in path:
                                        sent_embed_encoder=True
                                        sent_embed_decoder=True
                                        sent_embed_each=True
                                        sent_embed_postnet=True
                                        concat_sent_style=True
                                        use_concat_projection=True
                                    if "a10" in path:
                                        lang_embs = None
                                        utt_embed_dim = 192
                                        sent_embed_dim = None
                                    if "a11" in path:
                                        sent_embed_encoder=True
                                        concat_sent_style=True
                                        use_concat_projection=True
                                    if "a12" in path:
                                        sent_embed_encoder=True
                                        style_sent=True
                                        if "noadapt" in path and "adapted" not in path:
                                            utt_embed_dim = 768

                                    net = ToucanTTS(lang_embs=lang_embs, 
                                                    utt_embed_dim=utt_embed_dim,
                                                    sent_embed_dim=64 if "adapted" in path else sent_embed_dim,
                                                    sent_embed_adaptation="noadapt" not in path,
                                                    sent_embed_encoder=sent_embed_encoder,
                                                    sent_embed_decoder=sent_embed_decoder,
                                                    sent_embed_each=sent_embed_each,
                                                    sent_embed_postnet=sent_embed_postnet,
                                                    concat_sent_style=concat_sent_style,
                                                    use_concat_projection=use_concat_projection,
                                                    use_sent_style_loss="loss" in path,
                                                    pre_embed="_pre" in path,
                                                    style_sent=style_sent)
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
