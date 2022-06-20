import torch
import torch.multiprocessing
import torch.multiprocessing
from numpy import trim_zeros

from InferenceInterfaces.InferenceFastSpeech2 import InferenceFastSpeech2
from Preprocessing.AudioPreprocessor import AudioPreprocessor


class ProsodicConditionExtractor:

    def __init__(self, sr, model_id, device=torch.device("cpu")):
        self.ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024, cut_silence=False)
        self.tts = InferenceFastSpeech2(device=device, model_name=model_id)
        self.sr = sr
        self.device = device

    def extract_condition_from_reference_wave(self, wave, already_normalized=False):
        if already_normalized:
            norm_wave = wave.numpy()
        else:
            norm_wave = self.ap.audio_to_wave_tensor(normalize=True, audio=wave)
            norm_wave = trim_zeros(norm_wave.numpy())
        spec = self.ap.audio_to_mel_spec_tensor(norm_wave, explicit_sampling_rate=self.sr).transpose(0, 1)
        spec_len = torch.LongTensor([len(spec)])
        spec_batch = torch.stack([spec] * 5, dim=0)
        spec_len_batch = torch.stack([spec_len] * 5, dim=0)
        return torch.mean(self.tts.style_embedding_function(spec_batch.to(self.device),
                                                            spec_len_batch.to(self.device)), dim=0).squeeze()
