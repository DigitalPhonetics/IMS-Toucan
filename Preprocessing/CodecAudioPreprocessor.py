import torch

from torchaudio.transforms import Resample


class CodecAudioPreprocessor:

    def __init__(self, input_sr, output_sr=44100, device="cpu"):
        import dac  # have to do the imports down here, since it otherwise globally reserves GPU 0 instead of the correct one
        from dac.model import DAC
        from dac.utils import load_model
        self.device = device
        self.input_sr = input_sr
        self.output_sr = output_sr
        self.resample = Resample(orig_freq=input_sr, new_freq=output_sr).to(self.device)
        self.model = DAC()
        self.model = load_model(dac.__model_version__)
        self.model.eval()
        self.model.to(device)

    def resample_audio(self, audio, current_sampling_rate):
        if current_sampling_rate != self.input_sr:
            print("warning, change in sampling rate detected. If this happens too often, consider re-ordering the audios so that the sampling rate stays constant for multiple samples")
            self.resample = Resample(orig_freq=current_sampling_rate, new_freq=self.output_sr).to(self.device)
            self.input_sr = current_sampling_rate
        audio = torch.tensor(audio, device=self.device, dtype=torch.float32)
        audio = self.resample(audio)
        return audio

    @torch.inference_mode()
    def audio_to_codec_tensor(self, audio, current_sampling_rate):
        if current_sampling_rate != self.output_sr:
            audio = self.resample_audio(audio, current_sampling_rate)
        elif type(audio) != torch.tensor:
            audio = torch.Tensor(audio)
        return self.model.encode(audio.unsqueeze(0).unsqueeze(0))["z"].squeeze()

    @torch.inference_mode()
    def audio_to_codebook_indexes(self, audio, current_sampling_rate):
        if current_sampling_rate != self.output_sr:
            audio = self.resample_audio(audio, current_sampling_rate)
        elif type(audio) != torch.tensor:
            audio = torch.Tensor(audio)
        return self.model.encode(audio.unsqueeze(0).unsqueeze(0))["codes"].squeeze()

    @torch.inference_mode()
    def indexes_to_codec_frames(self, codebook_indexes):
        if len(codebook_indexes.size()) == 2:
            codebook_indexes = codebook_indexes.unsqueeze(0)
        return self.model.quantizer.from_codes(codebook_indexes)[0].squeeze()

    @torch.inference_mode()
    def codes_to_audio(self, continuous_codes):
        if len(continuous_codes.size()) == 2:
            continuous_codes = continuous_codes.unsqueeze(0)
        return self.model.decode(continuous_codes)["audio"].squeeze()


if __name__ == '__main__':
    import soundfile

    wav, sr = soundfile.read("../audios/96.wav")
    ap = CodecAudioPreprocessor(input_sr=sr)

    codebook_indexes = ap.audio_to_codebook_indexes(wav, current_sampling_rate=sr)
    continuous_codes_from_indexes = ap.indexes_to_codec_frames(codebook_indexes)

    reconstructed_audio = ap.codes_to_audio(continuous_codes_from_indexes).cpu().numpy()
    soundfile.write(file="../audios/96_reconstructed.wav", data=reconstructed_audio, samplerate=44100)
