import itertools
import os

import matplotlib.pyplot as plt
import pyloudnorm
import sounddevice
import soundfile
import torch
from speechbrain.pretrained import EncoderClassifier
from torchaudio.transforms import Resample

from Architectures.BigVGAN.BigVGAN import BigVGAN
from Architectures.EmbeddingModel.StyleEmbedding import StyleEmbedding
from Architectures.ToucanTTS.InferenceToucanTTS import ToucanTTS
from Preprocessing.AudioPreprocessor import AudioPreprocessor
from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
from Preprocessing.TextFrontend import get_language_id
from Utility.storage_config import MODELS_DIR
from Utility.utils import cumsum_durations
from Utility.utils import float2pcm


class ToucanTTSInterface(torch.nn.Module):

    def __init__(self,
                 device="cpu",  # device that everything computes on. If a cuda device is available, this can speed things up by an order of magnitude.
                 tts_model_path=os.path.join(MODELS_DIR, f"ToucanTTS_Meta", "best.pt"),  # path to the ToucanTTS checkpoint or just a shorthand if run standalone
                 vocoder_model_path=os.path.join(MODELS_DIR, f"BigVGAN", "best.pt"),  # path to the ToucanTTS checkpoint or just a shorthand if run standalone
                 embedding_model_path=None,
                 language="en",  # initial language of the model, can be changed later with the setter methods
                 ):
        super().__init__()
        self.device = device
        if not tts_model_path.endswith(".pt"):
            # default to shorthand system
            tts_model_path = os.path.join(MODELS_DIR, f"ToucanTTS_{tts_model_path}", "best.pt")

        ################################
        #   build text to phone        #
        ################################
        self.text2phone = ArticulatoryCombinedTextFrontend(language=language, add_silence_to_end=True)

        #####################################
        #   load phone to features model    #
        #####################################
        checkpoint = torch.load(tts_model_path, map_location='cpu')
        self.use_lang_id = True
        self.phone2codec = ToucanTTS(weights=checkpoint["model"], config=checkpoint["config"])  # multi speaker multi language
        with torch.no_grad():
            self.phone2codec.store_inverse_all()  # this also removes weight norm
        self.phone2codec = self.phone2codec.to(torch.device(device))

        ######################################
        #  load features to style models     #
        ######################################
        self.style_embedding_function = StyleEmbedding()
        if embedding_model_path is None:
            check_dict = torch.load(os.path.join(MODELS_DIR, "Embedding", "embedding_function.pt"), map_location="cpu")
        else:
            check_dict = torch.load(embedding_model_path, map_location="cpu")
        self.style_embedding_function.load_state_dict(check_dict["style_emb_func"])
        self.style_embedding_function.to(self.device)
        self.speaker_embedding_func_ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                                           run_opts={"device": str(device)},
                                                                           savedir=os.path.join(MODELS_DIR, "Embedding", "speechbrain_speaker_embedding_ecapa"))

        ################################
        #  load mel to wave model      #
        ################################
        vocoder_checkpoint = torch.load(vocoder_model_path, map_location="cpu")
        self.vocoder = BigVGAN()
        self.vocoder.load_state_dict(vocoder_checkpoint)
        self.vocoder = self.vocoder.to(device).eval()
        self.vocoder.remove_weight_norm()
        self.meter = pyloudnorm.Meter(24000)

        ################################
        #  set defaults                #
        ################################
        self.default_utterance_embedding = checkpoint["default_emb"].to(self.device)
        self.ap = AudioPreprocessor(input_sr=100, output_sr=16000, device=device)
        self.phone2codec.eval()
        self.vocoder.eval()
        self.style_embedding_function.eval()
        if self.use_lang_id:
            self.lang_id = get_language_id(language)
        else:
            self.lang_id = None
        self.to(torch.device(device))
        self.eval()

    def set_utterance_embedding(self, path_to_reference_audio="", embedding=None):
        if embedding is not None:
            self.default_utterance_embedding = embedding.squeeze().to(self.device)
            return
        assert os.path.exists(path_to_reference_audio)
        wave, sr = soundfile.read(path_to_reference_audio)
        wave = Resample(orig_freq=sr, new_freq=16000).to(self.device)(torch.tensor(wave, device=self.device, dtype=torch.float32))
        spec = self.ap.audio_to_mel_spec_tensor(wave, explicit_sampling_rate=16000)
        spec_len = torch.LongTensor([len(spec)])
        style_embedding = self.style_embedding_function(spec.unsqueeze(0).to(self.device), spec_len.unsqueeze(0).to(self.device)).squeeze()
        speaker_embedding = self.speaker_embedding_func_ecapa.encode_batch(wavs=wave.to(self.device).unsqueeze(0)).squeeze()
        self.default_utterance_embedding = torch.cat([style_embedding, speaker_embedding], dim=-1)

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
                input_is_phones=False,
                return_plot_as_filepath=False,
                loudness_in_db=-24.0):
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
            mel, durations, pitch, energy = self.phone2codec(phones,
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
            # codec_frames=self.codec_wrapper.model.quantizer(codec_frames.unsqueeze(0))[0].squeeze()  # re-quantization

            wave, _, _ = self.vocoder(mel.unsqueeze(0))
            wave = wave.squeeze().cpu().numpy()
        try:
            loudness = self.meter.integrated_loudness(wave)
            wave = pyloudnorm.normalize.loudness(wave, loudness, loudness_in_db)
        except ValueError:
            # if the audio is too short, a value error will arise
            pass

        if view or return_plot_as_filepath:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 5))

            ax.imshow(mel.cpu().numpy(), origin="lower", cmap='GnBu')
            ax.yaxis.set_visible(False)
            duration_splits, label_positions = cumsum_durations(durations.cpu().numpy())
            ax.xaxis.grid(True, which='minor')
            ax.set_xticks(label_positions, minor=False)
            if input_is_phones:
                phones = text.replace(" ", "|")
            else:
                phones = self.text2phone.get_phone_string(text, for_plot_labels=True)
            ax.set_xticklabels(phones)
            word_boundaries = list()
            for label_index, phone in enumerate(phones):
                if phone == "|":
                    word_boundaries.append(label_positions[label_index])

            try:
                prev_word_boundary = 0
                word_label_positions = list()
                for word_boundary in word_boundaries:
                    word_label_positions.append((word_boundary + prev_word_boundary) / 2)
                    prev_word_boundary = word_boundary
                word_label_positions.append((duration_splits[-1] + prev_word_boundary) / 2)

                secondary_ax = ax.secondary_xaxis('bottom')
                secondary_ax.tick_params(axis="x", direction="out", pad=24)
                secondary_ax.set_xticks(word_label_positions, minor=False)
                secondary_ax.set_xticklabels(text.split())
                secondary_ax.tick_params(axis='x', colors='orange')
                secondary_ax.xaxis.label.set_color('orange')
            except ValueError:
                ax.set_title(text)
            except IndexError:
                ax.set_title(text)

            ax.vlines(x=duration_splits, colors="green", linestyles="solid", ymin=0, ymax=120, linewidth=0.5)
            ax.vlines(x=word_boundaries, colors="orange", linestyles="solid", ymin=0, ymax=120, linewidth=1.0)
            plt.subplots_adjust(left=0.02, bottom=0.2, right=0.98, top=.9, wspace=0.0, hspace=0.0)
            ax.set_aspect("auto")

            if return_plot_as_filepath:
                plt.savefig("tmp.png")
                return wave, "tmp.png"
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
        silence = torch.zeros([14300])
        wav = silence.clone()
        for (text, durations, pitch, energy) in itertools.zip_longest(text_list, dur_list, pitch_list, energy_list):
            if text.strip() != "":
                if not silent:
                    print("Now synthesizing: {}".format(text))
                spoken_sentence = torch.tensor(self(text,
                                                    durations=durations.to(self.device) if durations is not None else None,
                                                    pitch=pitch.to(self.device) if pitch is not None else None,
                                                    energy=energy.to(self.device) if energy is not None else None,
                                                    duration_scaling_factor=duration_scaling_factor,
                                                    pitch_variance_scale=pitch_variance_scale,
                                                    energy_variance_scale=energy_variance_scale)).cpu()
                wav = torch.cat((wav, spoken_sentence, silence), 0)
        soundfile.write(file=file_location, data=float2pcm(wav), samplerate=24000, subtype="PCM_16")

    def read_aloud(self,
                   text,
                   view=False,
                   duration_scaling_factor=1.0,
                   pitch_variance_scale=1.0,
                   energy_variance_scale=1.0,
                   blocking=False):
        if text.strip() == "":
            return
        wav = torch.tensor(self(text,
                                view,
                                duration_scaling_factor=duration_scaling_factor,
                                pitch_variance_scale=pitch_variance_scale,
                                energy_variance_scale=energy_variance_scale))
        wav = torch.cat((torch.zeros([10000]), wav, torch.zeros([10000])), 0).numpy()
        sounddevice.play(float2pcm(wav), samplerate=24000)
        if view:
            plt.show()
        if blocking:
            sounddevice.wait()
