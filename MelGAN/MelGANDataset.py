import random

import soundfile as sf
import torch.nn.functional as F
from torch.utils.data import Dataset

from PreprocessingForTTS.ProcessAudio import AudioPreprocessor


class MelGANDataset(Dataset):

    def __init__(self, list_of_paths, samples_per_segment=4096):
        self.list_of_paths = list_of_paths
        self.ap = None
        self.samples_per_segment = samples_per_segment
        # has to be divisible by hop size. Selected for a 16kHz signal, as they did in the paper.

    def __getitem__(self, index):
        """
        load the audio from the path and clean it.
        All audio segments have to be cut or padded to the same length,
        according to the NeurIPS reference implementation.

        return a pair of cleaned audio and corresponding spectrogram
        """
        file_path = self.list_of_paths[index]
        wave, sr = sf.read(file_path)
        if self.ap is None:
            self.ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024)
            # hop length must be same as the product of the upscale factors
        normalized_wave = self.ap.audio_to_wave_tensor(wave, normalize=True, mulaw=False)
        if len(normalized_wave) <= self.samples_per_segment:
            # pad to size
            segment = F.pad(normalized_wave, (0, self.samples_per_segment - len(normalized_wave), "constant"))
        else:
            # cut to size, random segment
            max_audio_start = len(normalized_wave) - self.samples_per_segment
            audio_start = random.randint(0, max_audio_start)
            segment = normalized_wave[audio_start: audio_start + self.samples_per_segment]
        melspec = self.ap.audio_to_mel_spec_tensor(segment, normalize=False)
        print(len(melspec[0]))
        print(len(melspec[0] * 256))
        return segment, melspec

    def __len__(self):
        return len(self.list_of_paths)
