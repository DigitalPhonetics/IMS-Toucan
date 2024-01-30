import torch.multiprocessing

from Architectures.ToucanTTS.TTSDataset import TTSDataset
from Utility.path_to_transcript_dicts import *


def prepare_tts_corpus(path_list,
                       latents_list,
                       corpus_dir,
                       gpu_count=1,
                       rank=0):
    return TTSDataset(path_list,
                      latents_list,
                      cache_dir=corpus_dir,
                      device=torch.device("cuda"),
                      gpu_count=gpu_count,
                      rank=rank)
