import os

import librosa.display as lbd
import matplotlib.pyplot as plt
import sounddevice
import soundfile
import torch

from InferenceInterfaces.InferenceArchitectures.InferenceFastSpeech2 import FastSpeech2
from InferenceInterfaces.InferenceArchitectures.InferenceHiFiGAN import HiFiGANGenerator
from Preprocessing.ArticulatoryCombinedTextFrontend import ArticulatoryCombinedTextFrontend


class Nancy_FastSpeech2(torch.nn.Module):

    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.text2phone = ArticulatoryCombinedTextFrontend(language="en", inference=True)
        self.phone2mel = FastSpeech2(path_to_weights=os.path.join("Models", "FastSpeech2_Nancy", "best.pt")).to(torch.device(device))
        self.mel2wav = HiFiGANGenerator(path_to_weights=os.path.join("Models", "HiFiGAN_combined", "best.pt")).to(torch.device(device))
        self.phone2mel.eval()
        self.mel2wav.eval()
        self.to(torch.device(device))

    def forward(self, text, view=False):
        with torch.no_grad():
            phones = self.text2phone.string_to_tensor(text).to(torch.device(self.device))
            mel, durations, pitch, energy = self.phone2mel(phones, return_duration_pitch_energy=True)
            mel = mel.transpose(0, 1)
            wave = self.mel2wav(mel)
        if view:
            from Utility.utils import cumsum_durations
            fig, ax = plt.subplots(nrows=2, ncols=1)
            ax[0].plot(wave.cpu().numpy())
            lbd.specshow(mel.cpu().numpy(),
                         ax=ax[1],
                         sr=16000,
                         cmap='GnBu',
                         y_axis='mel',
                         x_axis=None,
                         hop_length=256)
            ax[0].yaxis.set_visible(False)
            ax[1].yaxis.set_visible(False)
            duration_splits, label_positions = cumsum_durations(durations.cpu().numpy())
            ax[1].set_xticks(duration_splits, minor=True)
            ax[1].xaxis.grid(True, which='minor')
            ax[1].set_xticks(label_positions, minor=False)
            ax[1].set_xticklabels(self.text2phone.get_phone_string(text))
            ax[0].set_title(text)
            plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=.9, wspace=0.0, hspace=0.0)
            plt.show()
        return wave

    def read_to_file(self, text_list, file_location, silent=False):
        """
        :param silent: Whether to be verbose about the process
        :param text_list: A list of strings to be read
        :param file_location: The path and name of the file it should be saved to
        """
        wav = None
        silence = torch.zeros([8000])
        for text in text_list:
            if text.strip() != "":
                if not silent:
                    print("Now synthesizing: {}".format(text))
                if wav is None:
                    wav = self(text).cpu()
                    wav = torch.cat((wav, silence), 0)
                else:
                    wav = torch.cat((wav, self(text).cpu()), 0)
                    wav = torch.cat((wav, silence), 0)
        soundfile.write(file=file_location, data=wav.cpu().numpy(), samplerate=16000)

    def read_aloud(self, text, view=False, blocking=False):
        if text.strip() == "":
            return
        wav = self(text, view).cpu()
        wav = torch.cat((wav, torch.zeros([8000])), 0)
        if not blocking:
            sounddevice.play(wav.numpy(), samplerate=16000)
        else:
            sounddevice.play(torch.cat((wav, torch.zeros([12000])), 0).numpy(), samplerate=16000)
            sounddevice.wait()

    def save_pretrained_weights(self):
        torch.save(self.phone2mel.encoder.state_dict(), "Models/PretrainedModelFast/enc.pt")
        torch.save(self.phone2mel.decoder.state_dict(), "Models/PretrainedModelFast/dec.pt")
        torch.save(self.phone2mel.pitch_predictor.state_dict(), "Models/PretrainedModelFast/pitch_predictor.pt")
        torch.save(self.phone2mel.energy_predictor.state_dict(), "Models/PretrainedModelFast/energy_predictor.pt")
        torch.save(self.phone2mel.duration_predictor.state_dict(), "Models/PretrainedModelFast/duration_predictor.pt")
        torch.save(self.phone2mel.feat_out.state_dict(), "Models/PretrainedModelFast/feat_out.pt")
        torch.save(self.phone2mel.postnet.state_dict(), "Models/PretrainedModelFast/postnet.pt")
