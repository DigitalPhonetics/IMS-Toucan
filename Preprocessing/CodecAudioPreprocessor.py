import dac
import torch
from audiotools import AudioSignal
from dac.model import DAC
from dac.utils import load_model
from dac.utils.encode import process as encode
from torchaudio.transforms import Resample


class AudioPreprocessor:

    def __init__(self, input_sr, output_sr=44100, device="cpu"):
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

    def audio_to_codec_tensor(self, audio, current_sampling_rate):
        if current_sampling_rate != self.output_sr:
            audio = self.resample_audio(audio, current_sampling_rate)
        return encode(AudioSignal(audio, sample_rate=self.output_sr, device=self.device), self.device, self.model)


if __name__ == '__main__':
    import soundfile

    wav, sr = soundfile.read("../audios/ad00_0004.wav")
    ap = AudioPreprocessor(input_sr=sr)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))

    plt.imshow(ap.audio_to_codec_tensor(wav, current_sampling_rate=sr)["codes"].cpu().numpy(), cmap='GnBu')
    plt.show()
