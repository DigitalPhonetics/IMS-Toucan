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


class HiFiGANDataset(Dataset):

    def __init__(self,
                 list_of_original_paths,
                 list_of_synthetic_paths,
                 desired_samplingrate=24000,
                 samples_per_segment=12288,  # = (8192 * 3) 2 , as I used 8192 for 16kHz previously
                 loading_processes=max(os.cpu_count() - 2, 1)):
        self.samples_per_segment = samples_per_segment
        self.desired_samplingrate = desired_samplingrate
        self.melspec_ap = AudioPreprocessor(input_sr=self.desired_samplingrate,
                                            output_sr=16000,
                                            cut_silence=False)
        # hop length of spec loss should be same as the product of the upscale factors
        # samples per segment must be a multiple of hop length of spec loss
        list_of_paths = list(zip(list_of_original_paths, list_of_synthetic_paths))
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
        print("{} eligible audios found".format(len(self.waves)))

    def cache_builder_process(self, path_split):
        for path in tqdm(path_split):
            try:
                path1, path2 = path

                wave1, sr = sf.read(path1)
                if len(wave1.shape) == 2:
                    wave1 = librosa.to_mono(numpy.transpose(wave1))
                if sr != self.desired_samplingrate:
                    wave1 = librosa.resample(y=wave1, orig_sr=sr, target_sr=self.desired_samplingrate)

                wave2, sr = sf.read(path2)
                if len(wave2.shape) == 2:
                    wave2 = librosa.to_mono(numpy.transpose(wave2))
                if sr != self.desired_samplingrate:
                    wave2 = librosa.resample(y=wave2, orig_sr=sr, target_sr=self.desired_samplingrate)

                self.waves.append((wave1, wave2))
            except RuntimeError:
                print(f"Problem with the following path: {path}")

    def __getitem__(self, index):
        """
        load the audio from the path and clean it.
        All audio segments have to be cut to the same length,
        according to the NeurIPS reference implementation.

        return a pair of high-res audio and corresponding low-res spectrogram as if it was predicted by the TTS
        """
        try:
            wave1 = self.waves[index][0]
            wave2 = self.waves[index][1]
            while len(wave1) < self.samples_per_segment + 50:  # + 50 is just to be extra sure
                # catch files that are too short to apply meaningful signal processing and make them longer
                wave1 = numpy.concatenate([wave1, numpy.zeros(shape=1000), wave1])
                wave2 = numpy.concatenate([wave2, numpy.zeros(shape=1000), wave2])
                # add some true silence in the mix, so the vocoder is exposed to that as well during training
            wave1 = torch.Tensor(wave1)
            wave2 = torch.Tensor(wave2)

            max_audio_start = len(wave1) - self.samples_per_segment
            audio_start = random.randint(0, max_audio_start)
            segment1 = wave1[audio_start: audio_start + self.samples_per_segment]
            segment2 = wave2[audio_start: audio_start + self.samples_per_segment]

            resampled_segment = self.melspec_ap.resample(segment2).float()  # 16kHz spectrogram as input, 24kHz wave as output, see Blizzard 2021 DelightfulTTS
            melspec = self.melspec_ap.audio_to_mel_spec_tensor(resampled_segment,
                                                               explicit_sampling_rate=16000,
                                                               normalize=False).transpose(0, 1)[:-1].transpose(0, 1)
            return segment1.detach(), melspec.detach()
        except RuntimeError:
            print("encountered a runtime error, using fallback strategy")
            if index == 0:
                index = len(self.waves) - 1
            return self.__getitem__(index - 1)

    def __len__(self):
        return len(self.waves)
