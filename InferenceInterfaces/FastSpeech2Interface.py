import itertools
import os

import librosa.display as lbd
import matplotlib.pyplot as plt
import sounddevice
import soundfile
import torch

from InferenceInterfaces.InferenceArchitectures.InferenceFastSpeech2 import FastSpeech2
from InferenceInterfaces.InferenceArchitectures.InferenceHiFiGAN import HiFiGANGenerator
from Preprocessing.AudioPreprocessor import AudioPreprocessor
from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
from Preprocessing.TextFrontend import get_language_id
from TrainingInterfaces.Spectrogram_to_Embedding.StyleEmbedding import StyleEmbedding


class InferenceFastSpeech2(torch.nn.Module):

    def __init__(self, device="cpu", model_name="Meta", language="en"):
        super().__init__()
        self.device = device

        ################################
        #   build text to phone        #
        ################################
        self.text2phone = ArticulatoryCombinedTextFrontend(language=language, add_silence_to_end=True)

        ################################
        #   load weights               #
        ################################
        checkpoint = torch.load(os.path.join("Models", f"FastSpeech2_{model_name}", "best.pt"), map_location='cpu')

        ################################
        #   load phone to mel model    #
        ################################
        self.use_lang_id = True
        try:
            self.phone2mel = FastSpeech2(weights=checkpoint["model"]).to(torch.device(device))  # multi speaker multi language
        except RuntimeError:
            try:
                self.use_lang_id = False
                self.phone2mel = FastSpeech2(weights=checkpoint["model"], lang_embs=None).to(torch.device(device))  # multi speaker single language
            except RuntimeError:
                self.phone2mel = FastSpeech2(weights=checkpoint["model"], lang_embs=None, utt_embed_dim=None).to(torch.device(device))  # single speaker

        #################################
        #  load mel to style models     #
        #################################
        self.style_embedding_function = StyleEmbedding()
        check_dict = torch.load("Models/Embedding/embedding_function.pt", map_location="cpu")
        self.style_embedding_function.load_state_dict(check_dict["style_emb_func"])
        self.style_embedding_function.to(self.device)

        ################################
        #  load mel to wave model      #
        ################################
        self.mel2wav = HiFiGANGenerator(path_to_weights=os.path.join("Models", "Avocodo", "best.pt")).to(torch.device(device))

        ################################
        #  set defaults                #
        ################################
        self.default_utterance_embedding = checkpoint["default_emb"].to(self.device)
        self.audio_preprocessor = AudioPreprocessor(input_sr=16000, output_sr=16000, cut_silence=True, device=self.device)
        self.phone2mel.eval()
        self.mel2wav.eval()
        if self.use_lang_id:
            self.lang_id = get_language_id(language)
        else:
            self.lang_id = None
        self.to(torch.device(device))

    def set_utterance_embedding(self, path_to_reference_audio="", embedding=None):
        if embedding is not None:
            self.default_utterance_embedding = embedding.squeeze().to(self.device)
            return
        assert os.path.exists(path_to_reference_audio)
        wave, sr = soundfile.read(path_to_reference_audio)
        if sr != self.audio_preprocessor.sr:
            self.audio_preprocessor = AudioPreprocessor(input_sr=sr, output_sr=16000, cut_silence=True, device=self.device)
        spec = self.audio_preprocessor.audio_to_mel_spec_tensor(wave).transpose(0, 1)
        spec_len = torch.LongTensor([len(spec)])
        self.default_utterance_embedding = self.style_embedding_function(spec.unsqueeze(0).to(self.device),
                                                                         spec_len.unsqueeze(0).to(self.device)).squeeze()

    def set_language(self, lang_id):
        """
        The id parameter actually refers to the shorthand. This has become ambiguous with the introduction of the actual language IDs
        """
        self.set_phonemizer_language(lang_id=lang_id)
        self.set_accent_language(lang_id=lang_id)

    def set_phonemizer_language(self, lang_id):
        self.text2phone = ArticulatoryCombinedTextFrontend(language=lang_id, add_silence_to_end=True)

    def set_accent_language(self, lang_id):
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
                pause_duration_scaling_factor=1.0,
                durations=None,
                pitch=None,
                energy=None,
                input_is_phones=False):
        """
        duration_scaling_factor: reasonable values are 0.8 < scale < 1.2.
                                     1.0 means no scaling happens, higher values increase durations for the whole
                                     utterance, lower values decrease durations for the whole utterance.
        pitch_variance_scale: reasonable values are 0.6 < scale < 1.4.
                                  1.0 means no scaling happens, higher values increase variance of the pitch curve,
                                  lower values decrease variance of the pitch curve.
        energy_variance_scale: reasonable values are 0.6 < scale < 1.4.
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
                                                           energy_variance_scale=energy_variance_scale,
                                                           pause_duration_scaling_factor=pause_duration_scaling_factor)
            mel = mel.transpose(0, 1)
            wave = self.mel2wav(mel)
        if view:
            from Utility.utils import cumsum_durations
            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(9, 6))
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
            ax[1].xaxis.grid(True, which='minor')
            ax[1].set_xticks(label_positions, minor=False)
            phones = self.text2phone.get_phone_string(text, for_plot_labels=True)
            ax[1].set_xticklabels(phones)
            word_boundaries = list()
            for label_index, word_boundary in enumerate(phones):
                if word_boundary == "|":
                    word_boundaries.append(label_positions[label_index])
            ax[1].vlines(x=duration_splits, colors="green", linestyles="dotted", ymin=0.0, ymax=8000, linewidth=1.0)
            ax[1].vlines(x=word_boundaries, colors="orange", linestyles="dotted", ymin=0.0, ymax=8000, linewidth=1.0)
            pitch_array = pitch.cpu().numpy()
            for pitch_index, xrange in enumerate(zip(duration_splits[:-1], duration_splits[1:])):
                if pitch_array[pitch_index] != 0:
                    ax[1].hlines(pitch_array[pitch_index] * 1000, xmin=xrange[0], xmax=xrange[1], color="blue", linestyles="solid", linewidth=0.5)
            ax[0].set_title(text)
            plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=.9, wspace=0.0, hspace=0.0)
            plt.show()
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
            duration_scaling_factor: reasonable values are 0.8 < scale < 1.2.
                                     1.0 means no scaling happens, higher values increase durations for the whole
                                     utterance, lower values decrease durations for the whole utterance.
            pitch_variance_scale: reasonable values are 0.6 < scale < 1.4.
                                  1.0 means no scaling happens, higher values increase variance of the pitch curve,
                                  lower values decrease variance of the pitch curve.
            energy_variance_scale: reasonable values are 0.6 < scale < 1.4.
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
