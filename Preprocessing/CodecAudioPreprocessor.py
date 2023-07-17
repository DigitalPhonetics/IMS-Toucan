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
            audio = torch.tensor(audio,device=self.device, dtype=torch.float32)
        return self.model.encode(audio.unsqueeze(0).unsqueeze(0))["z"].squeeze()

    @torch.inference_mode()
    def audio_to_codebook_indexes(self, audio, current_sampling_rate):
        if current_sampling_rate != self.output_sr:
            audio = self.resample_audio(audio, current_sampling_rate)
        elif type(audio) != torch.tensor:
            audio = torch.tensor(audio,device=self.device, dtype=torch.float32)
        return self.model.encode(audio.unsqueeze(0).unsqueeze(0))["codes"].squeeze()

    @torch.inference_mode()
    def audio_to_one_hot_indexes(self, audio, current_sampling_rate):
        if current_sampling_rate != self.output_sr:
            audio = self.resample_audio(audio, current_sampling_rate)
        elif type(audio) != torch.tensor:
            audio = torch.tensor(audio,device=self.device, dtype=torch.float32)
        return self.indexes_to_one_hot(self.model.encode(audio.unsqueeze(0).unsqueeze(0))["codes"].squeeze())

    @torch.inference_mode()
    def indexes_to_one_hot(self, indexes):
        return torch.nn.functional.one_hot(indexes.squeeze(), num_classes=self.model.quantizer.codebook_size)

    @torch.inference_mode()
    def indexes_to_codec_frames(self, codebook_indexes):
        if len(codebook_indexes.size()) == 2:
            codebook_indexes = codebook_indexes.unsqueeze(0)
        return self.model.quantizer.from_codes(codebook_indexes)[1].squeeze()

    @torch.inference_mode()
    def indexes_to_audio(self, codebook_indexes):
        return self.codes_to_audio(self.indexes_to_codec_frames(codebook_indexes))

    @torch.inference_mode()
    def indexes_to_continuous_codec_frames(self, codebook_indexes):
        if len(codebook_indexes.size()) == 2:
            codebook_indexes = codebook_indexes.unsqueeze(0)
        return self.model.quantizer.from_codes(codebook_indexes)[0].squeeze()

    @torch.inference_mode()
    def codes_to_audio(self, continuous_codes):
        z_q = 0.0
        z_ps = torch.split(continuous_codes, self.model.codebook_dim, dim=0)
        for i, z_p in enumerate(z_ps):
            z_q_i = self.model.quantizer.quantizers[i].out_proj(z_p)
            z_q = z_q + z_q_i
        return self.model.decode(z_q.unsqueeze(0))["audio"].squeeze()

    @torch.inference_mode()
    def continuous_codes_to_audio(self, continuous_codes):
        return self.model.decode(continuous_codes.transpose(0, 1).unsqueeze(0))["audio"].squeeze()


if __name__ == '__main__':
    import soundfile

    test_audio = "../audios/ad00_0004.wav"
    wav, sr = soundfile.read(test_audio)
    ap = CodecAudioPreprocessor(input_sr=sr)

    one_hot_indexes = ap.audio_to_one_hot_indexes(wav, current_sampling_rate=sr)

    import matplotlib.pyplot as plt

    for codebook_index, codebook in enumerate(one_hot_indexes):
        fix, ax = plt.subplots(1, 1)
        ax.imshow(codebook.cpu().transpose(0, 1).numpy(), cmap='binary')
        ax.set_aspect("auto")
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        plt.subplots_adjust(left=0., bottom=0., right=1., top=1., wspace=0.0, hspace=0.0)
        plt.show()
