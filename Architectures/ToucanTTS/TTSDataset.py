import os

import librosa
import soundfile as sf
import torch
import torch.multiprocessing
from speechbrain.pretrained import EncoderClassifier
from torch.utils.data import Dataset
from torchaudio.transforms import Resample

from Preprocessing.AudioPreprocessor import AudioPreprocessor
from Utility.storage_config import MODELS_DIR


class TTSDataset(Dataset):

    def __init__(self,
                 path_list,
                 latents_list,
                 device,
                 gpu_count=1,
                 rank=0):

        self.path_list = path_list
        self.latents_list = latents_list
        self.assumed_sr = 1
        self.gpu_count = gpu_count
        self.rank = rank
        self.resample = Resample(orig_freq=self.assumed_sr, new_freq=16000)
        self.spec_extractor_for_features = AudioPreprocessor(input_sr=16000, output_sr=16000)
        self.speaker_embedding_func = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                                     run_opts={"device": str(device)},
                                                                     savedir=os.path.join(MODELS_DIR, "Embedding", "speechbrain_speaker_embedding_ecapa"))

        if self.gpu_count > 1:
            # we only keep a chunk of the dataset in memory to avoid redundancy. Which chunk, we figure out using the rank.
            while len(self.path_list) % self.gpu_count != 0:
                self.path_list.pop(-1)  # a bit unfortunate, but if you're using multiple GPUs, you probably have a ton of datapoints anyway.
            chunksize = int(len(self.path_list) / self.gpu_count)
            self.path_list = self.path_list[chunksize * self.rank:chunksize * (self.rank + 1)]

    def __getitem__(self, index):

        wave, sr = sf.read(self.path_list[index])
        wave = librosa.to_mono(wave)
        if sr != self.assumed_sr:
            self.assumed_sr = sr
            self.resample = Resample(orig_freq=self.assumed_sr, new_freq=16000)
        norm_wave = self.resample(torch.tensor(wave).float())

        latent_input_sequence = self.latents_list[index]
        features = self.spec_extractor_for_features.audio_to_mel_spec_tensor(norm_wave, explicit_sampling_rate=16000)
        feature_lengths = torch.LongTensor([len(features[0])])

        speaker_embed = self.speaker_embedding_func.encode_batch(wavs=norm_wave).squeeze()

        return latent_input_sequence, \
               features, \
               feature_lengths, \
               speaker_embed

    def __len__(self):
        return len(self.path_list)
