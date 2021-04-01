import random
from multiprocessing import Process, Manager

import soundfile as sf
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from PreprocessingForTTS.ProcessAudio import AudioPreprocessor


class MelGANDataset(Dataset):

    def __init__(self,
                 list_of_paths,
                 samples_per_segment=10240,
                 loading_processes=6):
        self.samples_per_segment = samples_per_segment
        self.list_of_norm_wave_paths = list()
        _, sr = sf.read(list_of_paths[0])
        #  ^ this is the reason why we must create individual
        # datasets and then concat them. If we just did all
        # datasets at once, there could be multiple sampling
        # rates.
        self.ap = AudioPreprocessor(input_sr=sr, output_sr=None, melspec_buckets=80, hop_length=256, n_fft=1024)
        # hop length must be same as the product of the upscale factors
        ressource_manager = Manager()
        self.list_of_norm_wave_paths = ressource_manager.list()
        # make processes
        path_splits = list()
        process_list = list()
        for i in range(loading_processes):
            path_splits.append(list_of_paths[i * len(list_of_paths) // loading_processes:(i + 1) * len(
                list_of_paths) // loading_processes])
        for path_split in path_splits:
            process_list.append(
                Process(target=self.cache_builder_process, args=(path_split, samples_per_segment), daemon=True))
            process_list[-1].start()
        for process in process_list:
            process.join()
        self.list_of_norm_wave_paths = list(self.list_of_norm_wave_paths)
        print("{} eligible audios found".format(len(self.list_of_norm_wave_paths)))

    def cache_builder_process(self, path_split, samples_per_segment):
        for path in tqdm(path_split):
            wave, sr = sf.read(path)
            if (len(wave) / sr) > ((samples_per_segment + 50) / 16000):  # + 50 is just to be extra sure
                # catch files that are too short to apply meaningful signal processing
                self.list_of_norm_wave_paths.append(path)

    def __getitem__(self, index):
        """
        load the audio from the path and clean it.
        All audio segments have to be cut to the same length,
        according to the NeurIPS reference implementation.

        return a pair of cleaned audio and corresponding spectrogram
        """
        wave_orig, _ = sf.read(self.list_of_norm_wave_paths[index])
        wave = self.ap.audio_to_wave_tensor(wave_orig, normalize=True, mulaw=False)
        max_audio_start = len(wave) - self.samples_per_segment
        audio_start = random.randint(0, max_audio_start)
        segment = torch.Tensor(wave[audio_start: audio_start + self.samples_per_segment])
        melspec = self.ap.audio_to_mel_spec_tensor(segment, normalize=False).transpose(0, 1)[:-1].transpose(0, 1)
        return segment, melspec

    def __len__(self):
        return len(self.list_of_norm_wave_paths)
