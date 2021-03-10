import json
import os
import random

import soundfile as sf
from torch.utils.data import Dataset

from PreprocessingForTTS.ProcessAudio import AudioPreprocessor


class MelGANDataset(Dataset):

    def __init__(self, list_of_paths, samples_per_segment=8192, cache_dir=None):
        self.samples_per_segment = samples_per_segment
        if os.path.exists(cache_dir):
            # load cache
            with open(cache_dir, 'r') as fp:
                self.list_of_paths = json.load(fp)
            _, sr = sf.read(self.list_of_paths[0])
            self.ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024)
            # hop length must be same as the product of the upscale factors
            print("{} eligible audios found".format(len(self.list_of_paths)))
        else:
            # create cache
            file_path = list_of_paths[0]
            self.list_of_paths = list()
            _, sr = sf.read(file_path)
            self.ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024)
            # hop length must be same as the product of the upscale factors
            for path in list_of_paths:
                wave, sr = sf.read(file_path)
                norm_wave = self.ap.audio_to_wave_tensor(wave, normalize=True, mulaw=False)
                if len(norm_wave) > samples_per_segment:
                    self.list_of_paths.append(path)
            print("{} eligible audios found".format(len(self.list_of_paths)))
            with open(cache_dir, 'w') as fp:
                json.dump(self.list_of_paths, fp)

    def __getitem__(self, index):
        """
        load the audio from the path and clean it.
        All audio segments have to be cut or padded to the same length,
        according to the NeurIPS reference implementation.

        return a pair of cleaned audio and corresponding spectrogram
        """
        file_path = self.list_of_paths[index]
        wave, sr = sf.read(file_path)
        normalized_wave = self.ap.audio_to_wave_tensor(wave, normalize=True, mulaw=False)
        # cut to size, random segment
        max_audio_start = len(normalized_wave) - self.samples_per_segment
        audio_start = random.randint(0, max_audio_start)
        segment = normalized_wave[audio_start: audio_start + self.samples_per_segment]
        melspec = self.ap.audio_to_mel_spec_tensor(segment, normalize=False).transpose(0, 1)[:-1].transpose(0, 1)
        return segment, melspec

    def __len__(self):
        return len(self.list_of_paths)
