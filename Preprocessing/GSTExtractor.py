import os

import torch
import torch.multiprocessing
import torch.multiprocessing
from numpy import trim_zeros

from Preprocessing.AudioPreprocessor import AudioPreprocessor
from TrainingInterfaces.Spectrogram_to_Embedding.StyleEmbedding import StyleEmbedding
from Utility.storage_config import MODELS_DIR


class ProsodicConditionExtractor:

    def __init__(self, sr, device=torch.device("cpu"), path_to_model=os.path.join(MODELS_DIR, "Embedding", "embedding_function.pt")):
        self.ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024, cut_silence=False)
        self.embed = StyleEmbedding()
        check_dict = torch.load(path_to_model, map_location="cpu")
        self.embed.load_state_dict(check_dict["style_emb_func"])
        self.embed.to(device)
        self.sr = sr
        self.device = device

    def extract_condition_from_reference_wave(self, wave, already_normalized=False):
        if already_normalized:
            norm_wave = wave.numpy()
        else:
            norm_wave = self.ap.audio_to_wave_tensor(normalize=True, audio=wave)
            norm_wave = trim_zeros(norm_wave.numpy())
        spec = self.ap.audio_to_mel_spec_tensor(norm_wave, explicit_sampling_rate=self.sr).transpose(0, 1)
        spec_batch = torch.stack([spec] * 5, dim=0)
        spec_len_batch = torch.LongTensor([len(spec)] * 5)
        return torch.mean(self.embed(spec_batch.to(self.device), spec_len_batch.to(self.device)), dim=0).squeeze()
