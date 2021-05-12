import os
import random
import shutil

import torch

from FastSpeech2.FastSpeech2 import FastSpeech2
from FastSpeech2.FastSpeechDataset import FastSpeechDataset
from FastSpeech2.fastspeech2_train_loop import train_loop as fast_train_loop
from MelGAN.MelGANDataset import MelGANDataset
from MelGAN.MelGANGenerator import MelGANGenerator
from MelGAN.MelGANMultiScaleDiscriminator import MelGANMultiScaleDiscriminator
from MelGAN.melgan_train_loop import train_loop as train_loop_melgan
from TransformerTTS.TransformerTTS import Transformer
from TransformerTTS.TransformerTTSDataset import TransformerTTSDataset
from TransformerTTS.transformer_tts_train_loop import train_loop as trans_train_loop
from Utility.path_to_transcript_dicts import build_path_to_transcript_dict_ljspeech


def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))


def load_net_trans(path, idim=161, odim=80):
    check_dict = torch.load(path, map_location=torch.device("cpu"))
    net = Transformer(idim=idim, odim=odim, spk_embed_dim=None)
    net.load_state_dict(check_dict["model"])
    return net


def load_net_fast(path, idim=161, odim=80):
    check_dict = torch.load(path, map_location=torch.device("cpu"))
    net = FastSpeech2(idim=idim, odim=odim, spk_embed_dim=None)
    net.load_state_dict(check_dict["model"])
    return net


def load_net_trans_multi(path, idim=161, odim=80):
    check_dict = torch.load(path, map_location=torch.device("cpu"))
    net = Transformer(idim=idim, odim=odim, spk_embed_dim=256)
    net.load_state_dict(check_dict["model"])
    return net


def load_net_fast_multi(path, idim=161, odim=80):
    check_dict = torch.load(path, map_location=torch.device("cpu"))
    net = FastSpeech2(idim=idim, odim=odim, spk_embed_dim=256)
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


def save_model_for_use(model, name="Transformer_English_Single.pt", dict_name="model"):
    if model is None:
        return
    print("saving model...")
    torch.save({dict_name: model.state_dict()}, name)
    print("...done!")


def make_best_in_all(n=3):
    for model_dir in os.listdir("Models/IntegrationTest"):
        checkpoint_paths = get_n_recent_checkpoints_paths(checkpoint_dir="Models/IntegrationTest/{}".format(model_dir), n=n)
        averaged_model = average_checkpoints(checkpoint_paths, load_func=load_net_trans)
        save_model_for_use(model=averaged_model, name="Models/IntegrationTest/{}/best.pt".format(model_dir))


def run(gpu_id, resume_checkpoint, finetune, model_dir):
    if gpu_id == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")

    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
        device = torch.device("cuda")

    torch.manual_seed(13)
    random.seed(13)

    print("Preparing Transformer Data")

    cache_dir = os.path.join("Corpora", "IntegrationTest")
    save_dir = os.path.join("Models", "IntegrationTest", "trans")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path_to_transcript_dict = build_path_to_transcript_dict_ljspeech()
    remove_keys = list(path_to_transcript_dict.keys())[:-10:]
    for key in remove_keys:
        path_to_transcript_dict.pop(key)

    train_set_trans = TransformerTTSDataset(path_to_transcript_dict,
                                            cache_dir=cache_dir,
                                            lang="en",
                                            min_len_in_seconds=1,
                                            max_len_in_seconds=10,
                                            rebuild_cache=True,
                                            loading_processes=2)

    print("Training Transformer")

    model = Transformer(idim=161, odim=80, spk_embed_dim=None)

    trans_train_loop(net=model,
                     train_dataset=train_set_trans,
                     device=device,
                     save_directory=save_dir,
                     steps=20,
                     batch_size=4,
                     gradient_accumulation=1,
                     epochs_per_save=1,
                     use_speaker_embedding=False,
                     lang="en",
                     lr=0.001,
                     warmup_steps=8000,
                     path_to_checkpoint=resume_checkpoint,
                     fine_tune=finetune)

    make_best_in_all(n=5)

    print("Preparing FastSpeech Data")

    cache_dir = os.path.join("Corpora", "IntegrationTest")
    save_dir = os.path.join("Models", "IntegrationTest", "fast")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    acoustic_model = Transformer(idim=161, odim=80, spk_embed_dim=None)
    acoustic_model.load_state_dict(torch.load(os.path.join("Models", "IntegrationTest", "trans", "best.pt"),
                                              map_location='cpu')["model"])

    train_set_fast = FastSpeechDataset(path_to_transcript_dict,
                                       cache_dir=cache_dir,
                                       acoustic_model=acoustic_model,
                                       lang="en",
                                       min_len_in_seconds=1,
                                       max_len_in_seconds=10,
                                       device=device,
                                       rebuild_cache=True)

    model = FastSpeech2(idim=161, odim=80, spk_embed_dim=None)

    print("Training FastSpeech")

    fast_train_loop(net=model,
                    train_dataset=train_set_fast,
                    device=device,
                    save_directory=save_dir,
                    steps=20,
                    batch_size=2,
                    gradient_accumulation=1,
                    epochs_per_save=10,
                    use_speaker_embedding=False,
                    lang="en",
                    lr=0.001,
                    warmup_steps=8000,
                    path_to_checkpoint=resume_checkpoint,
                    fine_tune=finetune)

    print("Preparing MelGAN Data")

    model_save_dir = os.path.join("Models", "IntegrationTest", "melgan")
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    train_set_lj_melgan = MelGANDataset(list_of_paths=list(path_to_transcript_dict.keys()))
    generator = MelGANGenerator()
    generator.reset_parameters()
    multi_scale_discriminator = MelGANMultiScaleDiscriminator()

    print("Training MelGAN")

    train_loop_melgan(batch_size=2,
                      steps=20,
                      generator=generator,
                      discriminator=multi_scale_discriminator,
                      train_dataset=train_set_lj_melgan,
                      device=device,
                      generator_warmup_steps=10,
                      model_save_dir=model_save_dir,
                      path_to_checkpoint=resume_checkpoint)

    print("Integration Test ran successfully!\n"
          "Here is what has been produced:\n\n")

    list_files(os.path.join("Models", "IntegrationTest"))

    shutil.rmtree(os.path.join("Models", "IntegrationTest"))

    print("Cleanup is complete.")
