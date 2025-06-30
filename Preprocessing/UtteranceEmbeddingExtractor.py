import os

import torch
import torch.multiprocessing
from speechbrain.pretrained import EncoderClassifier
from torchaudio.transforms import Resample

from Modules.EmbeddingModel.StyleEmbedding import StyleEmbedding
from Preprocessing.HiFiCodecAudioPreprocessor import CodecAudioPreprocessor
from Utility.storage_config import MODELS_DIR


class ProsodicConditionExtractor:

    def __init__(self, device=torch.device("cpu"), path_to_model=os.path.join(MODELS_DIR, "Embedding", "embedding_function.pt")):
        self.ap = CodecAudioPreprocessor(input_sr=100, output_sr=2)
        self.embed = StyleEmbedding()
        check_dict = torch.load(path_to_model, map_location="cpu")
        self.embed.load_state_dict(check_dict["style_emb_func"])
        self.speaker_embedding_func_ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                                           run_opts={"device": str(device)},
                                                                           savedir=os.path.join(MODELS_DIR, "Embedding", "speechbrain_speaker_embedding_ecapa"))
        self.embed.to(device)
        self.device = device

    def extract_condition_from_reference_wave(self, wave, sr):
        wave_24khz = Resample(orig_freq=sr, new_freq=24000).to(self.device)(torch.tensor(wave, device=self.device, dtype=torch.float32))
        spec = self.ap.audio_to_codec_tensor(wave_24khz, current_sampling_rate=24000).transpose(0, 1)
        spec_len = torch.LongTensor([len(spec)])
        style_embedding = self.embed(spec.unsqueeze(0).to(self.device), spec_len.unsqueeze(0).to(self.device)).squeeze()
        wave_16kHz = Resample(orig_freq=sr, new_freq=16000).to(self.device)(torch.tensor(wave, device=self.device, dtype=torch.float32))
        speaker_embedding = self.speaker_embedding_func_ecapa.encode_batch(wavs=wave_16kHz.to(self.device).unsqueeze(0)).squeeze()
        return torch.cat([style_embedding, speaker_embedding], dim=-1)
