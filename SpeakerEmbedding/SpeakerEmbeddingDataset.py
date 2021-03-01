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
        speaker_1 = random.choice(self.speakers)

        self.purity_toggle = not self.purity_toggle
        if self.purity_toggle:
            path_1 = random.choice(self.speaker_to_paths[speaker_1])
            path_2 = random.choice(self.speaker_to_paths[speaker_1])
            while path_1 == path_2:
                path_2 = random.choice(self.speaker_to_paths[speaker_1])

        else:
            speaker_2 = random.choice(self.speakers)
            while speaker_2 == speaker_1:
                speaker_2 = random.choice(self.speakers)
            pass

        samp_1 = self.ap.audio_to_mel_spec_tensor(self.datapoints[item][0],
                                                  normalize=True).unsqueeze(0).unsqueeze(0)
        samp_2 = self.ap.audio_to_mel_spec_tensor(self.datapoints[item][1],
                                                  normalize=True).unsqueeze(0).unsqueeze(0)
        label = torch.Tensor(self.datapoints[item][2]).to(self.device)
        return samp_1, samp_2, label
