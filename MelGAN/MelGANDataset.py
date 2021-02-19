import soundfile as sf
import torch
from torch.utils.data import Dataset

from PreprocessingForTTS.ProcessAudio import AudioPreprocessor


class MelGANDataset(Dataset):

    def __init__(self, list_of_paths, device=torch.device("cpu"), type="train"):
        if type == "train":
            self.list_of_paths = list_of_paths[:-100]
        elif type == "valid":
            self.list_of_paths = list_of_paths[-100:]
        else:
            print("unknown set type ('train' or 'valid' are allowed)")
        self.device = device
        self.ap = None

    def __getitem__(self, index):
        # load the audio from the path, clean it, quantize it, process it into a spectrogram
        # return a pair of cleaned audio and spectrogram
        file_path = self.list_of_paths[index]
        wave, sr = sf.read(file_path)
        if self.ap is None:
            self.ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80)
        normalized_wave = self.ap.audio_to_wave_tensor(wave, normalize=True, mulaw=False).to(self.device)
        melspec = self.ap.audio_to_mel_spec_tensor(normalized_wave, normalize=False).to(self.device)
        return normalized_wave, melspec

    def __len__(self):
        return len(self.list_of_paths)
