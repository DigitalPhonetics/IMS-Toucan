import os
import random

import librosa
import soundfile as sf
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from Preprocessing.AudioPreprocessor import AudioPreprocessor
from Preprocessing.AudioPreprocessor import to_mono
from Preprocessing.TextFrontend import get_language_id
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeechDataset import FastSpeechDataset


class E2EDataset(Dataset):

    def __init__(self,
                 # tts related
                 path_to_transcript_dict,
                 acoustic_checkpoint_path,
                 cache_dir,
                 lang,
                 loading_processes=os.cpu_count() if os.cpu_count() is not None else 30,
                 min_len_in_seconds=1,
                 max_len_in_seconds=20,
                 cut_silence=False,
                 reduction_factor=1,
                 device=torch.device("cpu"),
                 rebuild_cache=False,
                 ctc_selection=True,
                 save_imgs=False,
                 # vocoder related
                 desired_samplingrate=24000,
                 samples_per_segment=12288,  # = (8192 * 3) 2 , as I used 8192 for 16kHz previously
                 ):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        if not os.path.exists(os.path.join(cache_dir, "fast_train_cache.pt")) or rebuild_cache:
            FastSpeechDataset(path_to_transcript_dict=path_to_transcript_dict,
                              acoustic_checkpoint_path=acoustic_checkpoint_path,
                              cache_dir=cache_dir,
                              lang=lang,
                              loading_processes=loading_processes,
                              min_len_in_seconds=min_len_in_seconds,
                              max_len_in_seconds=max_len_in_seconds,
                              cut_silence=cut_silence,
                              reduction_factor=reduction_factor,
                              device=device,
                              rebuild_cache=rebuild_cache,
                              ctc_selection=ctc_selection,
                              save_imgs=save_imgs)

        # just load the fastspeech datapoints from cache
        self.datapoints = torch.load(os.path.join(cache_dir, "fast_train_cache.pt"), map_location='cpu')
        self.language_id = get_language_id(lang)

        # create vocoder data
        list_of_paths = [d[-1] for d in self.datapoints]
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
        self.waves = list()
        self.wave_lens = list()
        for index, path in tqdm(enumerate(list_of_paths[::-1])):
            wave, sr = sf.read(path)
            wave = to_mono(wave)
            if (len(wave) / sr) < ((self.samples_per_segment + 50) / self.desired_samplingrate):
                # drop because too short
                self.datapoints.pop(len(list_of_paths) - index)
                continue
            if sr != self.desired_samplingrate:
                wave = librosa.resample(y=wave, orig_sr=sr, target_sr=self.desired_samplingrate)

            self.wave_lens.append(len(wave))
            self.waves.append(wave)

        print(f"Prepared an E2E dataset with {len(self.datapoints)}.")

    def get_random_window(self, real_wave, fake_wave):
        """
        pass as input a real wave and a fake wave.
        This will return a randomized but consistent window of each that can be passed to the discriminator
        """
        max_audio_start = len(real_wave) - self.samples_per_segment - 50
        audio_start = random.randint(0, max_audio_start)
        segment_real = real_wave[audio_start: audio_start + self.samples_per_segment]
        segment_fake = fake_wave[audio_start: audio_start + self.samples_per_segment]
        return segment_real, segment_fake

    def __getitem__(self, index):
        return self.datapoints[index][0], \
               self.datapoints[index][1], \
               self.datapoints[index][2], \
               self.datapoints[index][3], \
               self.datapoints[index][4], \
               self.datapoints[index][5], \
               self.datapoints[index][6], \
               self.datapoints[index][7], \
               self.language_id, \
               torch.Tensor(self.waves[index]), \
               torch.LongTensor([self.wave_lens[index]])

    def __len__(self):
        return len(self.datapoints)
