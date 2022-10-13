import math
import random

import librosa
import numpy
import soundfile as sf
import torch
from torch.utils.data import Dataset

from Preprocessing.AudioPreprocessor import AudioPreprocessor
from Preprocessing.AudioPreprocessor import to_mono


class HiFiGANDataset(Dataset):

    def __init__(self,
                 list_of_paths,
                 desired_samplingrate=48000,
                 samples_per_segment=24576,  # = 8192 * 3, as I used 8192 for 16kHz previously
                 use_random_corruption=False):
        self.use_random_corruption = use_random_corruption
        self.samples_per_segment = samples_per_segment
        self.desired_samplingrate = desired_samplingrate
        self.melspec_ap = AudioPreprocessor(input_sr=self.desired_samplingrate,
                                            output_sr=16000,
                                            melspec_buckets=80,
                                            hop_length=256,
                                            n_fft=1024,
                                            cut_silence=False)
        # hop length of spec loss should be same as the product of the upscale factors
        # samples per segment must be a multiple of hop length of spec loss
        self.paths = list_of_paths
        print("{} eligible audios found".format(len(self.paths)))

    def __getitem__(self, index):
        """
        load the audio from the path and clean it.
        All audio segments have to be cut to the same length,
        according to the NeurIPS reference implementation.

        return a pair of high-red audio and corresponding low-res spectrogram as if it was predicted by the TTS
        """
        path = self.paths[index]
        wave, sr = sf.read(path)  # this makes it so the disk is accessed way too much, but very efficient on the RAM
        wave = to_mono(wave)
        while (len(wave) / sr) < (
                (self.samples_per_segment + 50) / self.desired_samplingrate):  # + 50 is just to be extra sure
            # catch files that are too short to apply meaningful signal processing and make them longer
            wave = numpy.concatenate([wave, numpy.zeros(shape=1000), wave])
            # add some true silence in the mix, so the vocoder is exposed to that as well during training
        if sr == self.desired_samplingrate:
            wave = torch.tensor(wave)
        else:
            wave = torch.tensor(librosa.resample(y=wave, orig_sr=sr, target_sr=self.desired_samplingrate))

        max_audio_start = len(wave) - self.samples_per_segment
        audio_start = random.randint(0, max_audio_start)
        segment = wave[audio_start: audio_start + self.samples_per_segment]

        if random.random() < 0.1 and self.use_random_corruption:
            # apply distortion to random samples with a 10% chance
            noise = torch.rand(size=(segment.shape[0],)) - 0.5  # get 0 centered noise
            speech_power = segment.norm(p=2)
            noise_power = noise.norm(p=2)
            scale = math.sqrt(math.e) * noise_power / speech_power  # signal to noise ratio of 5db
            noisy_segment = (scale * segment + noise) / 2
            resampled_segment = self.melspec_ap.resample(
                noisy_segment)  # 16kHz spectrogram as input, 48kHz wave as output, see Blizzard 2021 DelightfulTTS
        else:
            resampled_segment = self.melspec_ap.resample(
                segment)  # 16kHz spectrogram as input, 48kHz wave as output, see Blizzard 2021 DelightfulTTS
        try:
            melspec = self.melspec_ap.audio_to_mel_spec_tensor(resampled_segment.float(),
                                                               explicit_sampling_rate=16000,
                                                               normalize=False).transpose(0, 1)[:-1].transpose(0, 1)
        except librosa.util.exceptions.ParameterError:
            # seems like sometimes adding noise and then resampling can introduce overflows which cause errors.
            melspec = self.melspec_ap.audio_to_mel_spec_tensor(self.melspec_ap.resample(segment).float(),
                                                               explicit_sampling_rate=16000,
                                                               normalize=False).transpose(0, 1)[:-1].transpose(0, 1)
        return segment, melspec

    def __len__(self):
        return len(self.paths)
