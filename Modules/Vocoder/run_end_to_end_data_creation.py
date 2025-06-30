"""
This script is meant to be executed from the top level of the repo to make all the paths resolve. It is just here for clean storage.
"""

import itertools

import librosa
import matplotlib.pyplot as plt
import sounddevice
import soundfile
import soundfile as sf
import torch
from speechbrain.pretrained import EncoderClassifier
from torchaudio.transforms import Resample
from tqdm import tqdm

from Modules.Aligner.Aligner import Aligner
from Modules.ToucanTTS.DurationCalculator import DurationCalculator
from Modules.ToucanTTS.EnergyCalculator import EnergyCalculator
from Modules.ToucanTTS.InferenceToucanTTS import ToucanTTS
from Modules.ToucanTTS.PitchCalculator import Parselmouth
from Preprocessing.AudioPreprocessor import AudioPreprocessor
from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
from Preprocessing.TextFrontend import get_language_id
from Preprocessing.articulatory_features import get_feature_to_index_lookup
from Utility.path_to_transcript_dicts import *
from Utility.storage_config import MODELS_DIR
from Utility.storage_config import PREPROCESSING_DIR
from Utility.utils import float2pcm


class ToucanTTSInterface(torch.nn.Module):

    def __init__(self,
                 device="cpu",  # device that everything computes on. If a cuda device is available, this can speed things up by an order of magnitude.
                 tts_model_path=os.path.join(MODELS_DIR, f"ToucanTTS_Meta", "best.pt"),  # path to the ToucanTTS checkpoint or just a shorthand if run standalone
                 vocoder_model_path=os.path.join(MODELS_DIR, f"Vocoder", "best.pt"),  # path to the Vocoder checkpoint
                 language="eng",  # initial language of the model, can be changed later with the setter methods
                 ):
        super().__init__()
        self.device = device
        if not tts_model_path.endswith(".pt"):
            tts_model_path = os.path.join(MODELS_DIR, f"ToucanTTS_{tts_model_path}", "best.pt")

        self.text2phone = ArticulatoryCombinedTextFrontend(language=language, add_silence_to_end=True, device=device)
        checkpoint = torch.load(tts_model_path, map_location='cpu')
        self.phone2mel = ToucanTTS(weights=checkpoint["model"], config=checkpoint["config"])
        with torch.no_grad():
            self.phone2mel.store_inverse_all()  # this also removes weight norm
        self.phone2mel = self.phone2mel.to(torch.device(device))
        self.speaker_embedding_func_ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                                           run_opts={"device": str(device)},
                                                                           savedir=os.path.join(MODELS_DIR, "Embedding", "speechbrain_speaker_embedding_ecapa"))
        self.default_utterance_embedding = checkpoint["default_emb"].to(self.device)
        self.ap = AudioPreprocessor(input_sr=100, output_sr=16000, device=device)
        self.phone2mel.eval()
        self.lang_id = get_language_id(language)
        self.to(torch.device(device))
        self.eval()

    def set_utterance_embedding(self, path_to_reference_audio="", embedding=None):
        if embedding is not None:
            self.default_utterance_embedding = embedding.squeeze().to(self.device)
            return
        if type(path_to_reference_audio) != list:
            path_to_reference_audio = [path_to_reference_audio]
        if len(path_to_reference_audio) > 0:
            for path in path_to_reference_audio:
                assert os.path.exists(path)
            speaker_embs = list()
            for path in path_to_reference_audio:
                wave, sr = soundfile.read(path)
                if len(wave.shape) > 1:  # oh no, we found a stereo audio!
                    if len(wave[0]) == 2:  # let's figure out whether we need to switch the axes
                        wave = wave.transpose()  # if yes, we switch the axes.
                wave = librosa.to_mono(wave)
                wave = Resample(orig_freq=sr, new_freq=16000).to(self.device)(torch.tensor(wave, device=self.device, dtype=torch.float32))
                speaker_embedding = self.speaker_embedding_func_ecapa.encode_batch(wavs=wave.to(self.device).squeeze().unsqueeze(0)).squeeze()
                speaker_embs.append(speaker_embedding)
            self.default_utterance_embedding = sum(speaker_embs) / len(speaker_embs)

    def set_language(self, lang_id):
        self.set_phonemizer_language(lang_id=lang_id)
        self.set_accent_language(lang_id=lang_id)

    def set_phonemizer_language(self, lang_id):
        self.text2phone = ArticulatoryCombinedTextFrontend(language=lang_id, add_silence_to_end=True, device=self.device)

    def set_accent_language(self, lang_id):
        if lang_id in {'ajp', 'ajt', 'lak', 'lno', 'nul', 'pii', 'plj', 'slq', 'smd', 'snb', 'tpw', 'wya', 'zua', 'en-us', 'en-sc', 'fr-be', 'fr-sw', 'pt-br', 'spa-lat', 'vi-ctr', 'vi-so'}:
            if lang_id == 'vi-so' or lang_id == 'vi-ctr':
                lang_id = 'vie'
            elif lang_id == 'spa-lat':
                lang_id = 'spa'
            elif lang_id == 'pt-br':
                lang_id = 'por'
            elif lang_id == 'fr-sw' or lang_id == 'fr-be':
                lang_id = 'fra'
            elif lang_id == 'en-sc' or lang_id == 'en-us':
                lang_id = 'eng'
            else:
                lang_id = 'eng'
        self.lang_id = get_language_id(lang_id).to(self.device)

    def forward(self,
                text,
                duration_scaling_factor=1.0,
                pitch_variance_scale=1.0,
                energy_variance_scale=1.0,
                pause_duration_scaling_factor=1.0,
                durations=None,
                pitch=None,
                energy=None,
                input_is_phones=False,
                prosody_creativity=0.1):
        with torch.inference_mode():
            phones = self.text2phone.string_to_tensor(text, input_phonemes=input_is_phones).to(torch.device(self.device))
            mel, _, _, _ = self.phone2mel(phones,
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
                                          prosody_creativity=prosody_creativity)
        return mel

    def read_to_file(self,
                     text_list,
                     file_location,
                     duration_scaling_factor=1.0,
                     pitch_variance_scale=1.0,
                     energy_variance_scale=1.0,
                     pause_duration_scaling_factor=1.0,
                     dur_list=None,
                     pitch_list=None,
                     energy_list=None,
                     prosody_creativity=0.1):
        if not dur_list:
            dur_list = []
        if not pitch_list:
            pitch_list = []
        if not energy_list:
            energy_list = []
        for (text, durations, pitch, energy) in itertools.zip_longest(text_list, dur_list, pitch_list, energy_list):
            spoken_sentence = self(text,
                                   durations=durations.to(self.device) if durations is not None else None,
                                   pitch=pitch.to(self.device) if pitch is not None else None,
                                   energy=energy.to(self.device) if energy is not None else None,
                                   duration_scaling_factor=duration_scaling_factor,
                                   pitch_variance_scale=pitch_variance_scale,
                                   energy_variance_scale=energy_variance_scale,
                                   pause_duration_scaling_factor=pause_duration_scaling_factor,
                                   prosody_creativity=prosody_creativity)
            spoken_sentence = spoken_sentence.cpu()

        torch.save(f=file_location, obj=spoken_sentence)

    def read_aloud(self,
                   text,
                   view=False,
                   duration_scaling_factor=1.0,
                   pitch_variance_scale=1.0,
                   energy_variance_scale=1.0,
                   blocking=False,
                   prosody_creativity=0.1):
        if text.strip() == "":
            return
        wav, sr = self(text,
                       view,
                       duration_scaling_factor=duration_scaling_factor,
                       pitch_variance_scale=pitch_variance_scale,
                       energy_variance_scale=energy_variance_scale,
                       prosody_creativity=prosody_creativity)
        silence = torch.zeros([sr // 2])
        wav = torch.cat((silence, torch.tensor(wav), silence), 0).numpy()
        sounddevice.play(float2pcm(wav), samplerate=sr)
        if view:
            plt.show()
        if blocking:
            sounddevice.wait()


class UtteranceCloner:

    def __init__(self, model_id, device, language="eng"):
        self.tts = ToucanTTSInterface(device=device, tts_model_path=model_id)
        self.ap = AudioPreprocessor(input_sr=100, output_sr=16000, cut_silence=False)
        self.tf = ArticulatoryCombinedTextFrontend(language=language, device=device)
        self.device = device
        acoustic_checkpoint_path = os.path.join(PREPROCESSING_DIR, "libri_all_clean", "Aligner", "aligner.pt")
        self.aligner_weights = torch.load(acoustic_checkpoint_path, map_location=device)["asr_model"]
        self.acoustic_model = Aligner()
        self.acoustic_model = self.acoustic_model.to(self.device)
        self.acoustic_model.load_state_dict(self.aligner_weights)
        self.acoustic_model.eval()
        self.parsel = Parselmouth(reduction_factor=1, fs=16000)
        self.energy_calc = EnergyCalculator(reduction_factor=1, fs=16000)
        self.dc = DurationCalculator(reduction_factor=1)

    def extract_prosody(self, transcript, ref_audio_path, lang="eng", on_line_fine_tune=False):
        wave, sr = sf.read(ref_audio_path)
        if self.tf.language != lang:
            self.tf = ArticulatoryCombinedTextFrontend(language=lang, device=self.device)
        if self.ap.input_sr != sr:
            self.ap = AudioPreprocessor(input_sr=sr, output_sr=16000, cut_silence=False)
        try:
            norm_wave = self.ap.normalize_audio(audio=wave)
        except ValueError:
            print('Something went wrong, the reference wave might be too short.')
            raise RuntimeError

        norm_wave_length = torch.LongTensor([len(norm_wave)])
        text = self.tf.string_to_tensor(transcript, handle_missing=False).squeeze(0)
        features = self.ap.audio_to_mel_spec_tensor(audio=norm_wave, explicit_sampling_rate=16000).transpose(0, 1)
        feature_length = torch.LongTensor([len(features)]).numpy()

        text_without_word_boundaries = list()
        indexes_of_word_boundaries = list()
        for phoneme_index, vector in enumerate(text):
            if vector[get_feature_to_index_lookup()["word-boundary"]] == 0:
                text_without_word_boundaries.append(vector.numpy().tolist())
            else:
                indexes_of_word_boundaries.append(phoneme_index)
        matrix_without_word_boundaries = torch.Tensor(text_without_word_boundaries)

        alignment_path = self.acoustic_model.inference(features=features.to(self.device),
                                                       tokens=matrix_without_word_boundaries.to(self.device),
                                                       return_ctc=False)

        duration = self.dc(torch.LongTensor(alignment_path), vis=None).cpu()

        for index_of_word_boundary in indexes_of_word_boundaries:
            duration = torch.cat([duration[:index_of_word_boundary],
                                  torch.LongTensor([0]),  # insert a 0 duration wherever there is a word boundary
                                  duration[index_of_word_boundary:]])

        energy = self.energy_calc(input_waves=norm_wave.unsqueeze(0),
                                  input_waves_lengths=norm_wave_length,
                                  feats_lengths=feature_length,
                                  text=text,
                                  durations=duration.unsqueeze(0),
                                  durations_lengths=torch.LongTensor([len(duration)]))[0].squeeze(0).cpu()
        pitch = self.parsel(input_waves=norm_wave.unsqueeze(0),
                            input_waves_lengths=norm_wave_length,
                            feats_lengths=feature_length,
                            text=text,
                            durations=duration.unsqueeze(0),
                            durations_lengths=torch.LongTensor([len(duration)]))[0].squeeze(0).cpu()
        return duration, pitch, energy

    def clone_utterance(self,
                        path_to_reference_audio_for_intonation,
                        path_to_reference_audio_for_voice,
                        transcription_of_intonation_reference,
                        filename_of_result=None,
                        lang="eng"):
        self.tts.set_utterance_embedding(path_to_reference_audio=path_to_reference_audio_for_voice)
        duration, pitch, energy = self.extract_prosody(transcription_of_intonation_reference,
                                                       path_to_reference_audio_for_intonation,
                                                       lang=lang)
        self.tts.set_language(lang)
        cloned_speech = self.tts(transcription_of_intonation_reference, view=False, durations=duration, pitch=pitch.transpose(0, 1), energy=energy.transpose(0, 1))
        if filename_of_result is not None:
            torch.save(f=filename_of_result, obj=cloned_speech)


class Reader:

    def __init__(self, language, device="cuda", model_id="Meta"):
        self.tts = UtteranceCloner(device=device, model_id=model_id, language=language)
        self.language = language

    def read_texts(self, sentence, filename, speaker_reference):
        self.tts.clone_utterance(speaker_reference,
                                 speaker_reference,
                                 sentence,
                                 filename_of_result=filename,
                                 lang=self.language)


if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

    all_dict = build_path_to_transcript_libritts_all_clean()

    reader = Reader(language="eng")
    for path in tqdm(all_dict):
        filename = path.replace(".wav", "_synthetic_spec.pt")
        reader.read_texts(sentence=all_dict[path], filename=filename, speaker_reference=path)
