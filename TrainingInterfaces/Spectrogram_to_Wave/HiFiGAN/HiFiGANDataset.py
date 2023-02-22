import math
import os
import random
from multiprocessing import Manager
from multiprocessing import Process

import librosa
import numpy
import soundfile as sf
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from Preprocessing.AudioPreprocessor import AudioPreprocessor
from Preprocessing.AudioPreprocessor import to_mono


class HiFiGANDataset(Dataset):

    def __init__(self,
                 list_of_paths,
                 desired_samplingrate=24000,
                 samples_per_segment=12288,  # = (8192 * 3) 2 , as I used 8192 for 16kHz previously
                 loading_processes=max(os.cpu_count() - 2, 1),
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
        if loading_processes == 1:
            self.waves = list()
            self.cache_builder_process(list_of_paths)
        else:
            resource_manager = Manager()
            self.waves = resource_manager.list()
            # make processes
            path_splits = list()
            process_list = list()
            for i in range(loading_processes):
                path_splits.append(list_of_paths[i * len(list_of_paths) // loading_processes:(i + 1) * len(
                    list_of_paths) // loading_processes])
            for path_split in path_splits:
                process_list.append(Process(target=self.cache_builder_process, args=(path_split,), daemon=True))
                process_list[-1].start()
            for process in process_list:
                process.join()
        self.waves = list(self.waves)
        print("{} eligible audios found".format(len(self.waves)))

    def cache_builder_process(self, path_split):
        for path in tqdm(path_split):
            try:
                wave, sr = sf.read(path)
                wave = to_mono(wave)
                if sr != self.desired_samplingrate:
                    wave = librosa.resample(y=wave, orig_sr=sr, target_sr=self.desired_samplingrate)
                self.waves.append(wave)
            except RuntimeError:
                print(f"Problem with the following path: {path}")

    def __getitem__(self, index):
        """
        load the audio from the path and clean it.
        All audio segments have to be cut to the same length,
        according to the NeurIPS reference implementation.

        return a pair of high-res audio and corresponding low-res spectrogram as if it was predicted by the TTS
        """
        wave = self.waves[index]
        while len(wave) < self.samples_per_segment + 50:  # + 50 is just to be extra sure
            # catch files that are too short to apply meaningful signal processing and make them longer
            wave = numpy.concatenate([wave, numpy.zeros(shape=1000), wave])
            # add some true silence in the mix, so the vocoder is exposed to that as well during training
        wave = torch.Tensor(wave)

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
                noisy_segment)  # 16kHz spectrogram as input, 24kHz wave as output, see Blizzard 2021 DelightfulTTS
        else:
            resampled_segment = self.melspec_ap.resample(
                segment)  # 16kHz spectrogram as input, 24kHz wave as output, see Blizzard 2021 DelightfulTTS
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
        return len(self.waves)
