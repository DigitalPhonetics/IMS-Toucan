"""
If feature extraction is updated, just call this to update all the caches to the new features

currently only the TransformerTTS caches are included.
"""

import os
import random
import warnings

import torch

from TransformerTTS.TransformerTTSDataset import TransformerTTSDataset
from Utility.path_to_transcript_dicts import build_path_to_transcript_dict_ljspeech, \
    build_path_to_transcript_dict_hokuspokus, build_path_to_transcript_dict_libritts

warnings.filterwarnings("ignore")

torch.manual_seed(13)
random.seed(13)


def rebuild_caches():
    # make sure cache directories exist
    cache_dir_lj = os.path.join("Corpora", "LJSpeech")
    if not os.path.exists(cache_dir_lj):
        os.makedirs(cache_dir_lj)
    cache_dir_libri = os.path.join("Corpora", "LibriTTS")
    if not os.path.exists(cache_dir_libri):
        os.makedirs(cache_dir_libri)
    cache_dir_css10de = os.path.join("Corpora", "CSS10_DE")
    if not os.path.exists(cache_dir_css10de):
        os.makedirs(cache_dir_css10de)

    # build all of the dicts that need to be cached
    path_to_transcript_dict_lj = build_path_to_transcript_dict_ljspeech()
    path_to_transcript_dict_libri = build_path_to_transcript_dict_libritts()
    path_to_transcript_dict_css10de = build_path_to_transcript_dict_hokuspokus()

    # remove any existing caches
    train_cache = "trans_train_cache.json"
    valid_cache = "trans_valid_cache.json"
    if os.path.exists(os.path.join(cache_dir_lj, train_cache)):
        os.remove(os.path.join(cache_dir_lj, train_cache))
    if os.path.exists(os.path.join(cache_dir_lj, valid_cache)):
        os.remove(os.path.join(cache_dir_lj, valid_cache))
    if os.path.exists(os.path.join(cache_dir_libri, train_cache)):
        os.remove(os.path.join(cache_dir_libri, train_cache))
    if os.path.exists(os.path.join(cache_dir_libri, valid_cache)):
        os.remove(os.path.join(cache_dir_libri, valid_cache))
    if os.path.exists(os.path.join(cache_dir_css10de, train_cache)):
        os.remove(os.path.join(cache_dir_css10de, train_cache))
    if os.path.exists(os.path.join(cache_dir_css10de, valid_cache)):
        os.remove(os.path.join(cache_dir_css10de, valid_cache))

    # finally build all the caches anew
    TransformerTTSDataset(path_to_transcript_dict_lj,
                          train=True,
                          cache_dir=cache_dir_lj,
                          lang="en",
                          min_len_in_seconds=0,
                          max_len_in_seconds=1000000)
    TransformerTTSDataset(path_to_transcript_dict_lj,
                          train=False,
                          cache_dir=cache_dir_lj,
                          lang="en",
                          min_len_in_seconds=0,
                          max_len_in_seconds=1000000)

    TransformerTTSDataset(path_to_transcript_dict_libri,
                          train=True,
                          cache_dir=cache_dir_libri,
                          lang="en",
                          min_len_in_seconds=10000,
                          max_len_in_seconds=400000,
                          spemb=True)
    TransformerTTSDataset(path_to_transcript_dict_libri,
                          train=False,
                          cache_dir=cache_dir_libri,
                          lang="en",
                          min_len_in_seconds=10000,
                          max_len_in_seconds=400000,
                          spemb=True)

    TransformerTTSDataset(path_to_transcript_dict_css10de,
                          train=True,
                          cache_dir=cache_dir_css10de,
                          lang="de",
                          min_len_in_seconds=0,
                          max_len_in_seconds=1000000)
    TransformerTTSDataset(path_to_transcript_dict_css10de,
                          train=False,
                          cache_dir=cache_dir_css10de,
                          lang="de",
                          min_len_in_seconds=0,
                          max_len_in_seconds=1000000)
