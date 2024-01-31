from Architectures.ToucanTTS.TTSDataset import TTSDataset


def prepare_tts_corpus(path_list,
                       latents_list,
                       gpu_count=1,
                       rank=0):
    return TTSDataset(path_list,
                      latents_list,
                      gpu_count=gpu_count,
                      rank=rank)
