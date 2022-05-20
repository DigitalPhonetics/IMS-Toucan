import itertools
import os

import librosa.display as lbd
import matplotlib.pyplot as plt
import noisereduce
import sounddevice
import soundfile
import torch

from InferenceInterfaces.InferenceArchitectures.InferenceFastSpeech2 import FastSpeech2
from InferenceInterfaces.InferenceArchitectures.InferenceHiFiGAN import HiFiGANGenerator
from Preprocessing.ProsodicConditionExtractor import ProsodicConditionExtractor
from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
from Preprocessing.TextFrontend import get_language_id


class InferenceFastSpeech2(torch.nn.Module):

    def __init__(self, device="cpu", model_name="Meta", language="en", noise_reduce=False):
        super().__init__()
        self.device = device
        self.text2phone = ArticulatoryCombinedTextFrontend(language=language, add_silence_to_end=True)
        checkpoint = torch.load(os.path.join("Models", f"FastSpeech2_{model_name}", "best.pt"), map_location='cpu')
        self.use_lang_id = True
        try:
            self.phone2mel = FastSpeech2(weights=checkpoint["model"]).to(torch.device(device))  # multi speaker multi language
        except RuntimeError:
            try:
                self.use_lang_id = False
                self.phone2mel = FastSpeech2(weights=checkpoint["model"], lang_embs=None).to(torch.device(device))  # multi speaker single language
            except RuntimeError:
                self.phone2mel = FastSpeech2(weights=checkpoint["model"], lang_embs=None, utt_embed_dim=None).to(torch.device(device))  # single speaker
        self.mel2wav = HiFiGANGenerator(path_to_weights=os.path.join("Models", "HiFiGAN_combined", "best.pt")).to(torch.device(device))
        self.default_utterance_embedding = checkpoint["default_emb"].to(self.device)
        self.phone2mel.eval()
        self.mel2wav.eval()
        if self.use_lang_id:
            self.lang_id = get_language_id(language)
        else:
            self.lang_id = None
        self.to(torch.device(device))
        self.noise_reduce = noise_reduce
        if self.noise_reduce:
            self.prototypical_noise = None
            self.update_noise_profile()

    def set_utterance_embedding(self, path_to_reference_audio):
        wave, sr = soundfile.read(path_to_reference_audio)
        self.default_utterance_embedding = ProsodicConditionExtractor(sr=sr).extract_condition_from_reference_wave(wave).to(self.device)
        if self.noise_reduce:
            self.update_noise_profile()

    def update_noise_profile(self):
        self.noise_reduce = False
        self.prototypical_noise = self("~." * 100, input_is_phones=True).cpu().numpy()
        self.noise_reduce = True

    def set_language(self, lang_id):
        """
        The id parameter actually refers to the shorthand. This has become ambiguous with the introduction of the actual language IDs
        """
        self.text2phone = ArticulatoryCombinedTextFrontend(language=lang_id, add_silence_to_end=True)
        if self.use_lang_id:
            self.lang_id = get_language_id(lang_id).to(self.device)
        else:
            self.lang_id = None

    def forward(self,
                text,
                view=False,
                duration_scaling_factor=1.0,
                pitch_variance_scale=1.0,
                energy_variance_scale=1.0,
                durations=None,
                pitch=None,
                energy=None,
                input_is_phones=False):
        """
        duration_scaling_factor: reasonable values are 0.5 < scale < 1.5.
                                     1.0 means no scaling happens, higher values increase durations for the whole
                                     utterance, lower values decrease durations for the whole utterance.
        pitch_variance_scale: reasonable values are 0.0 < scale < 2.0.
                                  1.0 means no scaling happens, higher values increase variance of the pitch curve,
                                  lower values decrease variance of the pitch curve.
        energy_variance_scale: reasonable values are 0.0 < scale < 2.0.
                                   1.0 means no scaling happens, higher values increase variance of the energy curve,
                                   lower values decrease variance of the energy curve.
        """
        with torch.inference_mode():
            phones = self.text2phone.string_to_tensor(text, input_phonemes=input_is_phones).to(torch.device(self.device))
            mel, durations, pitch, energy = self.phone2mel(phones,
                                                           return_duration_pitch_energy=True,
                                                           utterance_embedding=self.default_utterance_embedding,
                                                           durations=durations,
                                                           pitch=pitch,
                                                           energy=energy,
                                                           lang_id=self.lang_id,
                                                           duration_scaling_factor=duration_scaling_factor,
                                                           pitch_variance_scale=pitch_variance_scale,
                                                           energy_variance_scale=energy_variance_scale)
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
            ax[1].set_xticklabels(self.text2phone.get_phone_string(text, for_plot_labels=True))
            ax[0].set_title(text)
            plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=.9, wspace=0.0, hspace=0.0)
            plt.show()
        if self.noise_reduce:
            wave = torch.tensor(noisereduce.reduce_noise(y=wave.cpu().numpy(), y_noise=self.prototypical_noise, sr=48000, stationary=True), device=self.device)
        return wave

    def read_to_file(self,
                     text_list,
                     file_location,
                     duration_scaling_factor=1.0,
                     pitch_variance_scale=1.0,
                     energy_variance_scale=1.0,
                     silent=False,
                     dur_list=None,
                     pitch_list=None,
                     energy_list=None):
        """
        Args:
            silent: Whether to be verbose about the process
            text_list: A list of strings to be read
            file_location: The path and name of the file it should be saved to
            energy_list: list of energy tensors to be used for the texts
            pitch_list: list of pitch tensors to be used for the texts
            dur_list: list of duration tensors to be used for the texts
            duration_scaling_factor: reasonable values are 0.5 < scale < 1.5.
                                     1.0 means no scaling happens, higher values increase durations for the whole
                                     utterance, lower values decrease durations for the whole utterance.
            pitch_variance_scale: reasonable values are 0.0 < scale < 12.0.
                                  1.0 means no scaling happens, higher values increase variance of the pitch curve,
                                  lower values decrease variance of the pitch curve.
            energy_variance_scale: reasonable values are 0.0 < scale < 2.0.
                                   1.0 means no scaling happens, higher values increase variance of the energy curve,
                                   lower values decrease variance of the energy curve.
        """
        if not dur_list:
            dur_list = []
        if not pitch_list:
            pitch_list = []
        if not energy_list:
            energy_list = []
        wav = None
        silence = torch.zeros([24000])
        for (text, durations, pitch, energy) in itertools.zip_longest(text_list, dur_list, pitch_list, energy_list):
            if text.strip() != "":
                if not silent:
                    print("Now synthesizing: {}".format(text))
                if wav is None:
                    if durations is not None:
                        durations = durations.to(self.device)
                    if pitch is not None:
                        pitch = pitch.to(self.device)
                    if energy is not None:
                        energy = energy.to(self.device)
                    wav = self(text,
                               durations=durations,
                               pitch=pitch,
                               energy=energy,
                               duration_scaling_factor=duration_scaling_factor,
                               pitch_variance_scale=pitch_variance_scale,
                               energy_variance_scale=energy_variance_scale).cpu()
                    wav = torch.cat((wav, silence), 0)
                else:
                    wav = torch.cat((wav, self(text,
                                               durations=durations.to(self.device),
                                               pitch=pitch.to(self.device),
                                               energy=energy.to(self.device),
                                               duration_scaling_factor=duration_scaling_factor,
                                               pitch_variance_scale=pitch_variance_scale,
                                               energy_variance_scale=energy_variance_scale).cpu()), 0)
                    wav = torch.cat((wav, silence), 0)
        soundfile.write(file=file_location, data=wav.cpu().numpy(), samplerate=48000)

    def read_aloud(self,
                   text,
                   view=False,
                   duration_scaling_factor=1.0,
                   pitch_variance_scale=1.0,
                   energy_variance_scale=1.0,
                   blocking=False):
        if text.strip() == "":
            return
        wav = self(text,
                   view,
                   duration_scaling_factor=duration_scaling_factor,
                   pitch_variance_scale=pitch_variance_scale,
                   energy_variance_scale=energy_variance_scale).cpu()
        wav = torch.cat((wav, torch.zeros([24000])), 0)
        if not blocking:
            sounddevice.play(wav.numpy(), samplerate=48000)
        else:
            sounddevice.play(torch.cat((wav, torch.zeros([12000])), 0).numpy(), samplerate=48000)
            sounddevice.wait()
