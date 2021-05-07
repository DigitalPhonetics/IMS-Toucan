import os

import matplotlib.pyplot as plt
import sounddevice
import soundfile
import torch

from InferenceInterfaces.InferenceArchitectures.InferenceMelGAN import MelGANGenerator
from InferenceInterfaces.InferenceArchitectures.InferenceTransformerTTS import Transformer
from PreprocessingForTTS.ProcessText import TextFrontend


class Eva_TransformerTTSInference(torch.nn.Module):

    def __init__(self, device="cpu", speaker_embedding=None):
        super().__init__()
        self.speaker_embedding = None
        self.device = device
        self.text2phone = TextFrontend(language="de", use_word_boundaries=False, use_explicit_eos=False)
        self.phone2mel = Transformer(path_to_weights=os.path.join("Models", "TransformerTTS_Eva", "best.pt"),
                                     idim=123, odim=80, spk_embed_dim=None, reduction_factor=1).to(torch.device(device))
        self.mel2wav = MelGANGenerator(path_to_weights=os.path.join("Models", "MelGAN_combined", "best.pt")).to(torch.device(device))
        self.phone2mel.eval()
        self.mel2wav.eval()
        self.to(torch.device(device))

    def forward(self, text, view=False):
        with torch.no_grad():
            phones = self.text2phone.string_to_tensor(text).squeeze(0).long().to(torch.device(self.device))
            mel = self.phone2mel(phones, speaker_embedding=self.speaker_embedding).transpose(0, 1)
            wave = self.mel2wav(mel.unsqueeze(0)).squeeze(0).squeeze(0)
        if view:
            import matplotlib.pyplot as plt
            import librosa.display as lbd
            fig, ax = plt.subplots(nrows=2, ncols=1)
            ax[0].plot(wave.cpu().numpy())
            lbd.specshow(mel.cpu().numpy(), ax=ax[1], sr=16000, cmap='GnBu', y_axis='mel', x_axis='time', hop_length=256)
            ax[0].set_title(self.text2phone.get_phone_string(text))
            ax[0].yaxis.set_visible(False)
            ax[1].yaxis.set_visible(False)
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

    def plot_attentions(self, sentence):
        sentence_tensor = self.text2phone.string_to_tensor(sentence).squeeze(0).long().to(torch.device(self.device))
        att_ws = self.phone2mel(text=sentence_tensor, speaker_embedding=self.speaker_embedding, return_atts=True)
        atts = torch.cat([att_w for att_w in att_ws], dim=0)
        fig, axes = plt.subplots(nrows=len(atts) // 2, ncols=2, figsize=(6, 8))
        atts_1 = atts[::2]
        atts_2 = atts[1::2]
        for index, att in enumerate(atts_1):
            axes[index][0].imshow(att.detach().numpy(), interpolation='nearest', aspect='auto', origin="lower")
            axes[index][0].set_title("{}".format(index * 2))
            axes[index][0].xaxis.set_visible(False)
            axes[index][0].yaxis.set_visible(False)
        for index, att in enumerate(atts_2):
            axes[index][1].imshow(att.detach().numpy(), interpolation='nearest', aspect='auto', origin="lower")
            axes[index][1].set_title("{}".format((index + 1) * 2 - 1))
            axes[index][1].xaxis.set_visible(False)
            axes[index][1].yaxis.set_visible(False)
        plt.tight_layout()
        plt.show()
