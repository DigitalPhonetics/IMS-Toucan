import soundfile as sf
import torch
import torch.multiprocessing
import torch.multiprocessing
from numpy import trim_zeros
from speechbrain.pretrained import EncoderClassifier

from Preprocessing.AudioPreprocessor import AudioPreprocessor


class ProsodicConditionExtractor:

    def __init__(self, sr, device=torch.device("cpu")):
        self.ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024, cut_silence=False)
        # https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
        self.speaker_embedding_func_ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                                           run_opts={"device": str(device)},
                                                                           savedir="Models/SpeakerEmbedding/speechbrain_speaker_embedding_ecapa")
        # https://huggingface.co/speechbrain/spkrec-xvect-voxceleb
        self.speaker_embedding_func_xvector = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb",
                                                                             run_opts={"device": str(device)},
                                                                             savedir="Models/SpeakerEmbedding/speechbrain_speaker_embedding_xvector")

    def extract_condition_from_reference_wave(self, wave, already_normalized=False):
        if already_normalized:
            norm_wave = wave
        else:
            norm_wave = self.ap.audio_to_wave_tensor(normalize=True, audio=wave)
            norm_wave = torch.tensor(trim_zeros(norm_wave.numpy()))
        spk_emb_ecapa = self.speaker_embedding_func_ecapa.encode_batch(wavs=norm_wave.unsqueeze(0)).squeeze()
        spk_emb_xvector = self.speaker_embedding_func_xvector.encode_batch(wavs=norm_wave.unsqueeze(0)).squeeze()
        combined_utt_condition = torch.cat([spk_emb_ecapa.cpu(),
                                            spk_emb_xvector.cpu()], dim=0)
        return combined_utt_condition


if __name__ == '__main__':
    wave, sr = sf.read("../audios/1.wav")
    ext = ProsodicConditionExtractor(sr=sr)
    print(ext.extract_condition_from_reference_wave(wave=wave).shape)
