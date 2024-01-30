import os

import librosa
import soundfile as sf
import torch
from torch.utils.data import Dataset
from torchaudio.transforms import Resample
from tqdm import tqdm

from Preprocessing.AudioPreprocessor import AudioPreprocessor
from Preprocessing.EnCodecAudioPreprocessor import CodecAudioPreprocessor


class TTSDataset(Dataset):

    def __init__(self,
                 path_list,
                 latents_list,
                 cache_dir,
                 device=torch.device("cpu"),
                 rebuild_cache=False,
                 gpu_count=1,
                 rank=0):
        self.cache_dir = cache_dir
        self.device = device
        os.makedirs(cache_dir, exist_ok=True)
        if not os.path.exists(os.path.join(cache_dir, "tts_train_cache.pt")) or rebuild_cache:
            self._build_dataset_cache(path_list=path_list,
                                      latents_list=latents_list,
                                      cache_dir=cache_dir,
                                      device=device,
                                      gpu_count=gpu_count)
        self.cache_dir = cache_dir
        self.gpu_count = gpu_count
        self.rank = rank
        self.datapoints = torch.load(os.path.join(self.cache_dir, "tts_train_cache.pt"), map_location='cpu')
        if self.gpu_count > 1:
            # we only keep a chunk of the dataset in memory to avoid redundancy. Which chunk, we figure out using the rank.
            while len(self.datapoints) % self.gpu_count != 0:
                self.datapoints.pop(-1)  # a bit unfortunate, but if you're using multiple GPUs, you probably have a ton of datapoints anyway.
            chunksize = int(len(self.datapoints) / self.gpu_count)
            self.datapoints = self.datapoints[chunksize * self.rank:chunksize * (self.rank + 1)]
        print(f"Loaded a TTS dataset with {len(self.datapoints)} datapoints from {cache_dir}.")

    def _build_dataset_cache(self,
                             path_list,
                             latents_list,
                             cache_dir,
                             device=torch.device("cpu"),
                             gpu_count=1):
        if gpu_count != 1:
            import sys
            print("Please run the feature extraction using only a single GPU. Multi-GPU is only supported for training.")
            sys.exit()

        print("... building dataset cache ...")
        self.codec_wrapper = CodecAudioPreprocessor(input_sr=-1, device=device)
        self.spec_extractor_for_features = AudioPreprocessor(input_sr=16000, output_sr=16000, device=device)
        self.datapoints = list()

        # ==========================================
        # actual creation of datapoints starts here
        # ==========================================

        assumed_sr = 1
        ap = CodecAudioPreprocessor(input_sr=assumed_sr, device=device)
        resample = Resample(orig_freq=assumed_sr, new_freq=16000).to(device)

        for index, path in tqdm(enumerate(path_list)):
            try:
                wave, sr = sf.read(path)
            except:
                print(f"Problem with an audio file: {path}")
                continue

            wave = librosa.to_mono(wave)

            if sr != assumed_sr:
                assumed_sr = sr
                ap = CodecAudioPreprocessor(input_sr=assumed_sr, device=device)
                resample = Resample(orig_freq=assumed_sr, new_freq=16000).to(device)
                print(f"{path} has a different sampling rate --> adapting the codec processor")

            try:
                norm_wave = resample(torch.tensor(wave).float().to(device))
            except ValueError:
                continue

            codes = ap.audio_to_codebook_indexes(audio=norm_wave, current_sampling_rate=16000).transpose(0, 1).cpu().numpy()

            if codes.size()[0] != 24:  # no clue why this is sometimes the case
                codes = codes.transpose(0, 1)
            decoded_wave = self.codec_wrapper.indexes_to_audio(codes.int().to(device))
            features = self.spec_extractor_for_features.audio_to_mel_spec_tensor(decoded_wave, explicit_sampling_rate=16000)
            feature_lengths = torch.LongTensor([len(features[0])])

            text = latents_list[index]

            self.datapoints.append([text,  # text tensor
                                    codes,  # codec tensor (in index form)
                                    feature_lengths,  # length of spectrogram
                                    ])

        # =============================
        # done with datapoint creation
        # =============================

        # save to cache
        if len(self.datapoints) > 0:
            torch.save(self.datapoints, os.path.join(cache_dir, "tts_train_cache.pt"))
        else:
            import sys
            print("No datapoints were prepared! Exiting...")
            sys.exit()

    def __getitem__(self, index):
        return self.datapoints[index][0], \
            self.datapoints[index][1], \
            self.datapoints[index][2]

    def __len__(self):
        return len(self.datapoints)

    def remove_samples(self, list_of_samples_to_remove):
        for remove_id in sorted(list_of_samples_to_remove, reverse=True):
            self.datapoints.pop(remove_id)
        torch.save(self.datapoints, os.path.join(self.cache_dir, "tts_train_cache.pt"))
        print("Dataset updated!")
