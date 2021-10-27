import random
from multiprocessing import Manager
from multiprocessing import Process

import soundfile as sf
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from Preprocessing.AudioPreprocessor import AudioPreprocessor


class HiFiGANDataset(Dataset):

    def __init__(self,
                 list_of_paths,
                 desired_samplingrate=48000,
                 loading_processes=20):
        self.samples_per_segment = round(8192 * (desired_samplingrate / 16000))
        _, sr = sf.read(list_of_paths[0])
        #  ^ this is the reason why we must create individual
        # datasets and then concat them. If we just did all
        # datasets at once, there could be multiple sampling
        # rates.
        self.desired_samplingrate = desired_samplingrate
        self.preprocess_ap = AudioPreprocessor(input_sr=sr, output_sr=desired_samplingrate, melspec_buckets=80, hop_length=256, n_fft=1024, cut_silence=False)
        self.melspec_ap = AudioPreprocessor(input_sr=desired_samplingrate, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024, cut_silence=False)
        # hop length must be same as the product of the upscale factors
        resource_manager = Manager()
        self.list_of_eligible_wave_paths = resource_manager.list()
        # make processes
        path_splits = list()
        process_list = list()
        for i in range(loading_processes):
            path_splits.append(list_of_paths[i * len(list_of_paths) // loading_processes:(i + 1) * len(list_of_paths) // loading_processes])
        for path_split in path_splits:
            process_list.append(Process(target=self.cache_builder_process, args=(path_split,), daemon=True))
            process_list[-1].start()
        for process in process_list:
            process.join()
        self.list_of_eligible_wave_paths = list(self.list_of_eligible_wave_paths)
        self.waves = list()
        for path in tqdm(self.list_of_eligible_wave_paths):
            with open(path, "rb") as audio_file:
                wave_orig, _ = sf.read(audio_file)
            self.waves.append(self.preprocess_ap.audio_to_wave_tensor(wave_orig, normalize=True, mulaw=False))
        print("{} eligible audios found".format(len(self.waves)))

    def cache_builder_process(self, path_split):
        for path in tqdm(path_split):
            with open(path, "rb") as audio_file:
                wave, sr = sf.read(audio_file)
            if (len(wave) / sr) > ((self.samples_per_segment + 50) / self.desired_samplingrate):  # + 50 is just to be extra sure
                # catch files that are too short to apply meaningful signal processing
                self.list_of_eligible_wave_paths.append(path)

    def __getitem__(self, index):
        """
        load the audio from the path and clean it.
        All audio segments have to be cut to the same length,
        according to the NeurIPS reference implementation.

        return a pair of cleaned audio and corresponding spectrogram as if it was predicted by the TTS
        """
        max_audio_start = len(self.waves[index]) - self.samples_per_segment
        audio_start = random.randint(0, max_audio_start)
        segment = torch.Tensor(self.waves[index][audio_start: audio_start + self.samples_per_segment])
        resampled_segment = self.melspec_ap.resample(segment)
        melspec = self.melspec_ap.audio_to_mel_spec_tensor(resampled_segment, explicit_sampling_rate=16000, normalize=False).transpose(0, 1)[:-1].transpose(0,
                                                                                                                                                            1)
        return segment, melspec

    def __len__(self):
        return len(self.list_of_eligible_wave_paths)
