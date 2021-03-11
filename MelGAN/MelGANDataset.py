import json
import os
import random

import soundfile as sf
import torch
from torch.utils.data import Dataset

from PreprocessingForTTS.ProcessAudio import AudioPreprocessor


class MelGANDataset(Dataset):

    def __init__(self, list_of_paths, samples_per_segment=8192, cache_dir=None):
        self.samples_per_segment = samples_per_segment
        if os.path.exists(cache_dir):
            # load cache
            with open(cache_dir, 'r') as fp:
                self.list_of_norm_waves = json.load(fp)
            self.ap = AudioPreprocessor(input_sr=16000, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024)
            # hop length must be same as the product of the upscale factors
            print("{} eligible audios found".format(len(self.list_of_norm_waves)))
        else:
            # create cache
            file_path = list_of_paths[0]
            self.list_of_norm_waves = list()
            _, sr = sf.read(file_path)
            self.ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024)
            # hop length must be same as the product of the upscale factors
            for path in list_of_paths:
                wave, sr = sf.read(path)
                if len(wave) > 5000:
                    # catch files that are too short to apply meaningful signal processing
                    norm_wave = self.ap.audio_to_wave_tensor(wave, normalize=True, mulaw=False)
                    if len(norm_wave) > samples_per_segment:
                        self.list_of_norm_waves.append(norm_wave.detach().numpy().tolist())
            print("{} eligible audios found".format(len(self.list_of_norm_waves)))
            with open(cache_dir, 'w') as fp:
                json.dump(self.list_of_norm_waves, fp)

    def __getitem__(self, index):
        """
        load the audio from the path and clean it.
        All audio segments have to be cut to the same length,
        according to the NeurIPS reference implementation.

        return a pair of cleaned audio and corresponding spectrogram
        """
        max_audio_start = len(self.list_of_norm_waves[index]) - self.samples_per_segment
        audio_start = random.randint(0, max_audio_start)
        segment = torch.Tensor(self.list_of_norm_waves[index][audio_start: audio_start + self.samples_per_segment])
        melspec = self.ap.audio_to_mel_spec_tensor(segment, normalize=False).transpose(0, 1)[:-1].transpose(0, 1)
        return segment, melspec

    def __len__(self):
        return len(self.list_of_norm_waves)
