import os
import random

import soundfile as sf
import torch
from torch.utils.data import IterableDataset

from PreprocessingForTTS.ProcessAudio import AudioPreprocessor


class SpeakerEmbeddingDataset(IterableDataset):
    def __init__(self, train=True):
        if train:
            path_to_raw_corpus = "/mount/arbeitsdaten46/projekte/dialog-1/tillipl/" \
                                 "datasets/VoxCeleb2/audio-files/train/dev/aac/"
        else:
            path_to_raw_corpus = "/mount/arbeitsdaten46/projekte/dialog-1/tillipl/" \
                                 "datasets/VoxCeleb2/audio-files/test/aac/"

        self.speaker_to_paths = dict()
        for speaker in os.listdir(path_to_raw_corpus):
            self.speaker_to_paths[speaker] = list()
            for sub in os.listdir(os.path.join(path_to_raw_corpus, speaker)):
                for wav in os.listdir(os.path.join(path_to_raw_corpus, speaker, sub)):
                    if ".wav" in wav:
                        self.speaker_to_paths[speaker].append(os.path.join(path_to_raw_corpus, speaker, sub, wav))

        self.speakers = list(self.speaker_to_paths.keys())
        _, sr = sf.read(self.speaker_to_paths[self.speakers[0]])
        self.ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024)

        self.purity_toggle = True

    def __iter__(self):
        """
        provide two samples of the same speaker and two samples from different speakers in an alternating fashion
        """
        speaker_1 = random.choice(self.speakers)

        self.purity_toggle = not self.purity_toggle
        if self.purity_toggle:
            path_1 = random.choice(self.speaker_to_paths[speaker_1])
            wave_1 = None
            while wave_1 is None:
                try:
                    wave_1, _ = sf.read(path_1)
                    if len(wave_1) < 10000:
                        wave_1 = None
                except RuntimeError:
                    print("File {} seems to be faulty".format(path_1))
                    wave_1 = None
                    path_1 = random.choice(self.speaker_to_paths[speaker_1])
            path_2 = random.choice(self.speaker_to_paths[speaker_1])
            wave_2 = None
            while path_1 == path_2:
                path_2 = random.choice(self.speaker_to_paths[speaker_1])
            while wave_2 is None:
                try:
                    wave_2, _ = sf.read(path_2)
                    if len(wave_2) < 10000:
                        wave_2 = None
                except RuntimeError:
                    print("File {} seems to be faulty".format(path_2))
                    wave_2 = None
                    path_2 = random.choice(self.speaker_to_paths[speaker_1])
                    while path_1 == path_2:
                        path_2 = random.choice(self.speaker_to_paths[speaker_1])
            label = torch.IntTensor([-1])
        else:
            speaker_2 = random.choice(self.speakers)
            while speaker_2 == speaker_1:
                speaker_2 = random.choice(self.speakers)
            path_1 = random.choice(self.speaker_to_paths[speaker_1])
            wave_1 = None
            while wave_1 is None:
                try:
                    wave_1, _ = sf.read(path_1)
                    if len(wave_1) < 10000:
                        wave_1 = None
                except RuntimeError:
                    print("File {} seems to be faulty".format(path_1))
                    wave_1 = None
                    path_1 = random.choice(self.speaker_to_paths[speaker_1])
            path_2 = random.choice(self.speaker_to_paths[speaker_2])
            wave_2 = None
            while wave_2 is None:
                try:
                    wave_2, _ = sf.read(path_2)
                    if len(wave_2) < 10000:
                        wave_2 = None
                except RuntimeError:
                    print("File {} seems to be faulty".format(path_2))
                    wave_2 = None
                    path_2 = random.choice(self.speaker_to_paths[speaker_2])
            label = torch.IntTensor([1])

        data_1 = self.ap.audio_to_mel_spec_tensor(wave_1, normalize=True).unsqueeze(0).unsqueeze(0)
        data_2 = self.ap.audio_to_mel_spec_tensor(wave_2, normalize=True).unsqueeze(0).unsqueeze(0)
        return data_1, data_2, label
