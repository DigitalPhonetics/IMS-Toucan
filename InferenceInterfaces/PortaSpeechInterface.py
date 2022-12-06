import itertools
import os

import librosa.display as lbd
import matplotlib.pyplot as plt
import numpy
import pyloudnorm as pyln
import sounddevice
import soundfile
import torch
from df.enhance import enhance
from df.enhance import init_df
from pedalboard import Compressor
from pedalboard import HighShelfFilter
from pedalboard import HighpassFilter
from pedalboard import LowpassFilter
from pedalboard import NoiseGate
from pedalboard import PeakFilter
from pedalboard import Pedalboard

from InferenceInterfaces.InferenceArchitectures.InferenceAvocodo import HiFiGANGenerator
from InferenceInterfaces.InferenceArchitectures.InferencePortaSpeech import PortaSpeech
from Preprocessing.AudioPreprocessor import AudioPreprocessor
from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
from Preprocessing.TextFrontend import get_language_id
from TrainingInterfaces.Spectrogram_to_Embedding.StyleEmbedding import StyleEmbedding
from Utility.storage_config import MODELS_DIR


class PortaSpeechInterface(torch.nn.Module):

    def __init__(self,
                 device="cpu",
                 # device that everything computes on. If a cuda device is available, this can speed things up by an order of magnitude.
                 tts_model_path=os.path.join(MODELS_DIR, f"PortaSpeech_Meta", "best.pt"),
                 # path to the PortaSpeech checkpoint or just a shorthand if run standalone
                 vocoder_model_path=None,
                 # path to the hifigan/avocodo checkpoint
                 language="en",
                 # initial language of the model, can be changed later with the setter methods
                 use_enhancement=False,
                 # if you are using very low quality training data, you can use this to post-process your output
                 use_signalprocessing=False,
                 # some subtle effects that are frequently used in podcasting
                 use_post_glow=True
                 # whether to use the PostNet, as it is only trained after a large amount of steps, so it might still be random in preliminary checkpoints
                 ):
        super().__init__()
        self.device = device
        if not tts_model_path.endswith(".pt"):
            # default to shorthand system
            tts_model_path = os.path.join(MODELS_DIR, f"PortaSpeech_{tts_model_path}", "best.pt")
        if vocoder_model_path is not None:
            vocoder_model_path = os.path.join(MODELS_DIR, "Avocodo", "best.pt")
        self.use_signalprocessing = use_signalprocessing
        if self.use_signalprocessing:
            self.effects = Pedalboard(plugins=[HighpassFilter(cutoff_frequency_hz=60),
                                               HighShelfFilter(cutoff_frequency_hz=8000, gain_db=5.0),
                                               LowpassFilter(cutoff_frequency_hz=17000),
                                               PeakFilter(cutoff_frequency_hz=150, gain_db=2.0),
                                               PeakFilter(cutoff_frequency_hz=220, gain_db=-2.0),
                                               PeakFilter(cutoff_frequency_hz=900, gain_db=-2.0),
                                               PeakFilter(cutoff_frequency_hz=3200, gain_db=-2.0),
                                               PeakFilter(cutoff_frequency_hz=7500, gain_db=-2.0),
                                               NoiseGate(),
                                               Compressor(ratio=2.0)])
        self.use_enhancement = use_enhancement
        if self.use_enhancement:
            self.enhancer, self.df, _ = init_df(log_file=None,
                                                log_level="NONE",
                                                config_allow_defaults=True,
                                                post_filter=True)
            self.enhancer = self.enhancer.to(self.device).eval()
            self.loudnorm_meter = pyln.Meter(24000, block_size=0.200)

        ################################
        #   build text to phone        #
        ################################
        self.text2phone = ArticulatoryCombinedTextFrontend(language=language, add_silence_to_end=True)

        ################################
        #   load weights               #
        ################################
        checkpoint = torch.load(tts_model_path, map_location='cpu')

        ################################
        #   load phone to mel model    #
        ################################
        self.use_lang_id = True
        try:
            self.phone2mel = PortaSpeech(weights=checkpoint["model"], glow_enabled=use_post_glow)  # multi speaker multi language
        except RuntimeError:
            try:
                self.use_lang_id = False
                self.phone2mel = PortaSpeech(weights=checkpoint["model"],
                                             lang_embs=None,
                                             glow_enabled=use_post_glow)  # multi speaker single language
            except RuntimeError:
                self.phone2mel = PortaSpeech(weights=checkpoint["model"],
                                             lang_embs=None,
                                             utt_embed_dim=None,
                                             glow_enabled=use_post_glow)  # single speaker
        with torch.no_grad():
            self.phone2mel.store_inverse_all()
        self.phone2mel = self.phone2mel.to(torch.device(device))

        #################################
        #  load mel to style models     #
        #################################
        self.style_embedding_function = StyleEmbedding()
        check_dict = torch.load(os.path.join(MODELS_DIR, "Embedding", "embedding_function.pt"), map_location="cpu")
        self.style_embedding_function.load_state_dict(check_dict["style_emb_func"])
        self.style_embedding_function.to(self.device)

        ################################
        #  load mel to wave model      #
        ################################
        self.mel2wav = torch.jit.trace(HiFiGANGenerator(path_to_weights=vocoder_model_path), torch.rand((80, 50))).to(torch.device(device))

        ################################
        #  set defaults                #
        ################################
        self.default_utterance_embedding = checkpoint["default_emb"].to(self.device)
        self.audio_preprocessor = AudioPreprocessor(input_sr=16000, output_sr=16000, cut_silence=True, device=self.device)
        self.phone2mel.eval()
        self.mel2wav.eval()
        self.style_embedding_function.eval()
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
                                                           pause_duration_scaling_factor=pause_duration_scaling_factor,
                                                           device=self.device)
            mel = mel.transpose(0, 1)
            wave = self.mel2wav(mel)
            if self.use_signalprocessing:
                try:
                    wave = torch.Tensor(self.effects(wave.cpu().numpy(), 24000))
                except ValueError:
                    # if the audio is too short, a value error might arise
                    pass
            if self.use_enhancement:
                wave = enhance(self.enhancer, self.df, wave.unsqueeze(0).cpu(), pad=True).squeeze()
                try:
                    loudness = self.loudnorm_meter.integrated_loudness(wave.cpu().numpy())
                    loud_normed = pyln.normalize.loudness(wave.cpu().numpy(), loudness, -30.0)
                    peak = numpy.amax(numpy.abs(loud_normed))
                    wave = torch.Tensor(numpy.divide(loud_normed, peak))
                except ValueError:
                    # if the audio is too short, a value error will arise
                    pass

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
                    ax[1].hlines(pitch_array[pitch_index] * 1000, xmin=xrange[0], xmax=xrange[1], color="blue",
                                 linestyles="solid", linewidth=0.5)
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
                    wav = self(text,
                               durations=durations.to(self.device) if durations is not None else None,
                               pitch=pitch.to(self.device) if pitch is not None else None,
                               energy=energy.to(self.device) if energy is not None else None,
                               duration_scaling_factor=duration_scaling_factor,
                               pitch_variance_scale=pitch_variance_scale,
                               energy_variance_scale=energy_variance_scale).cpu()
                    wav = torch.cat((wav, silence), 0)
                else:
                    wav = torch.cat((wav, self(text,
                                               durations=durations.to(self.device) if durations is not None else None,
                                               pitch=pitch.to(self.device) if pitch is not None else None,
                                               energy=energy.to(self.device) if energy is not None else None,
                                               duration_scaling_factor=duration_scaling_factor,
                                               pitch_variance_scale=pitch_variance_scale,
                                               energy_variance_scale=energy_variance_scale).cpu()), 0)
                    wav = torch.cat((wav, silence), 0)
        soundfile.write(file=file_location, data=wav.cpu().numpy(), samplerate=24000)

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
        wav = torch.cat((wav, torch.zeros([12000])), 0)
        if not blocking:
            sounddevice.play(wav.numpy(), samplerate=24000)
        else:
            sounddevice.play(torch.cat((wav, torch.zeros([6000])), 0).numpy(), samplerate=24000)
            sounddevice.wait()
