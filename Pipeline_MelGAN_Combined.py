"""
Train non-autoregressive spectrogram inversion model on a combination of multiple large datasets

Spectrogram inversion is language and speaker independent,
so throwing together all datasets gives the best results.

"""

import gc
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import warnings

import torch
from torch.utils.data import ConcatDataset

from MelGAN.MelGANDataset import MelGANDataset
from MelGAN.MelGANGenerator import MelGANGenerator
from MelGAN.MelGANMultiScaleDiscriminator import MelGANMultiScaleDiscriminator
from MelGAN.melgan_train_loop import train_loop
from Utility.file_lists import get_file_list_ljspeech, get_file_list_css10ge, get_file_list_css10gr, \
    get_file_list_css10es, get_file_list_css10fi, get_file_list_css10ru, get_file_list_css10hu, get_file_list_css10du, \
    get_file_list_css10jp, get_file_list_css10ch, get_file_list_css10fr, get_file_list_thorsten

warnings.filterwarnings("ignore")

torch.manual_seed(13)
random.seed(13)

if __name__ == '__main__':
    print("Preparing")
    model_save_dir = "Models/MelGAN/MultiSpeaker/Combined_reworked"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    melgan_cache_dir = "Corpora/MelGAN"
    if not os.path.exists(melgan_cache_dir):
        os.makedirs(melgan_cache_dir)

    train_set_lj = MelGANDataset(list_of_paths=get_file_list_ljspeech()[:-50],
                                 cache=os.path.join(melgan_cache_dir, "LJSpeech_train.txt"))
    valid_set_lj = MelGANDataset(list_of_paths=get_file_list_ljspeech()[-50:],
                                 cache=os.path.join(melgan_cache_dir, "LJSpeech_valid.txt"))

    train_set_css10ge = MelGANDataset(list_of_paths=get_file_list_css10ge()[:-50],
                                      cache=os.path.join(melgan_cache_dir, "german_train.txt"))
    valid_set_css10ge = MelGANDataset(list_of_paths=get_file_list_css10ge()[-50:],
                                      cache=os.path.join(melgan_cache_dir, "german_valid.txt"))

    train_set_css10gr = MelGANDataset(list_of_paths=get_file_list_css10gr()[:-50],
                                      cache=os.path.join(melgan_cache_dir, "greek_train.txt"))
    valid_set_css10gr = MelGANDataset(list_of_paths=get_file_list_css10gr()[-50:],
                                      cache=os.path.join(melgan_cache_dir, "greek_valid.txt"))

    train_set_css10es = MelGANDataset(list_of_paths=get_file_list_css10es()[:-50],
                                      cache=os.path.join(melgan_cache_dir, "spanish_train.txt"))
    valid_set_css10es = MelGANDataset(list_of_paths=get_file_list_css10es()[-50:],
                                      cache=os.path.join(melgan_cache_dir, "spanish_valid.txt"))

    train_set_css10fi = MelGANDataset(list_of_paths=get_file_list_css10fi()[:-50],
                                      cache=os.path.join(melgan_cache_dir, "finnish_train.txt"))
    valid_set_css10fi = MelGANDataset(list_of_paths=get_file_list_css10fi()[-50:],
                                      cache=os.path.join(melgan_cache_dir, "finnish_valid.txt"))

    train_set_css10ru = MelGANDataset(list_of_paths=get_file_list_css10ru()[:-50],
                                      cache=os.path.join(melgan_cache_dir, "russian_train.txt"))
    valid_set_css10ru = MelGANDataset(list_of_paths=get_file_list_css10ru()[-50:],
                                      cache=os.path.join(melgan_cache_dir, "russian_valid.txt"))

    train_set_css10hu = MelGANDataset(list_of_paths=get_file_list_css10hu()[:-50],
                                      cache=os.path.join(melgan_cache_dir, "hungarian_train.txt"))
    valid_set_css10hu = MelGANDataset(list_of_paths=get_file_list_css10hu()[-50:],
                                      cache=os.path.join(melgan_cache_dir, "hungarian_valid.txt"))

    train_set_css10du = MelGANDataset(list_of_paths=get_file_list_css10du()[:-50],
                                      cache=os.path.join(melgan_cache_dir, "dutch_train.txt"))
    valid_set_css10du = MelGANDataset(list_of_paths=get_file_list_css10du()[-50:],
                                      cache=os.path.join(melgan_cache_dir, "dutch_valid.txt"))

    train_set_css10jp = MelGANDataset(list_of_paths=get_file_list_css10jp()[:-50],
                                      cache=os.path.join(melgan_cache_dir, "japanese_train.txt"))
    valid_set_css10jp = MelGANDataset(list_of_paths=get_file_list_css10jp()[-50:],
                                      cache=os.path.join(melgan_cache_dir, "japanese_valid.txt"))

    train_set_css10ch = MelGANDataset(list_of_paths=get_file_list_css10ch()[:-50],
                                      cache=os.path.join(melgan_cache_dir, "chinese_train.txt"))
    valid_set_css10ch = MelGANDataset(list_of_paths=get_file_list_css10ch()[-50:],
                                      cache=os.path.join(melgan_cache_dir, "chinese_valid.txt"))

    train_set_css10fr = MelGANDataset(list_of_paths=get_file_list_css10fr()[:-50],
                                      cache=os.path.join(melgan_cache_dir, "french_train.txt"))
    valid_set_css10fr = MelGANDataset(list_of_paths=get_file_list_css10fr()[-50:],
                                      cache=os.path.join(melgan_cache_dir, "french_valid.txt"))

    train_set_thorsten = MelGANDataset(list_of_paths=get_file_list_thorsten()[:-50],
                                       cache=os.path.join(melgan_cache_dir, "thorsten_train.txt"))
    valid_set_thorsten = MelGANDataset(list_of_paths=get_file_list_thorsten()[-50:],
                                       cache=os.path.join(melgan_cache_dir, "thorsten_valid.txt"))

    train_set = ConcatDataset([train_set_lj,
                               train_set_css10ge,
                               train_set_css10gr,
                               train_set_css10es,
                               train_set_css10fi,
                               train_set_css10ru,
                               train_set_css10hu,
                               train_set_css10du,
                               train_set_css10jp,
                               train_set_css10ch,
                               train_set_css10fr,
                               train_set_thorsten])
    valid_set = ConcatDataset([valid_set_lj,
                               valid_set_css10ge,
                               valid_set_css10gr,
                               valid_set_css10es,
                               valid_set_css10fi,
                               valid_set_css10ru,
                               valid_set_css10hu,
                               valid_set_css10du,
                               valid_set_css10jp,
                               valid_set_css10ch,
                               valid_set_css10fr,
                               valid_set_thorsten])

    gc.collect()

    generator = MelGANGenerator()
    generator.reset_parameters()
    multi_scale_discriminator = MelGANMultiScaleDiscriminator()

    print("Training model")
    train_loop(batchsize=16,
               steps=2000000,
               generator=generator,
               discriminator=multi_scale_discriminator,
               train_dataset=train_set,
               valid_dataset=valid_set,
               device=torch.device("cuda"),
               generator_warmup_steps=100000,
               model_save_dir=model_save_dir)
