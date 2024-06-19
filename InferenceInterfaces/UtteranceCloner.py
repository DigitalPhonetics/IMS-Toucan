import os

import numpy
import soundfile as sf
import torch

from Architectures.Aligner.Aligner import Aligner
from Architectures.ToucanTTS.DurationCalculator import DurationCalculator
from Architectures.ToucanTTS.EnergyCalculator import EnergyCalculator
from Architectures.ToucanTTS.PitchCalculator import Parselmouth
from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface
from Preprocessing.AudioPreprocessor import AudioPreprocessor
from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
from Preprocessing.articulatory_features import get_feature_to_index_lookup
from Utility.storage_config import MODELS_DIR
from Utility.utils import float2pcm


class UtteranceCloner:
    """
    Clone the prosody of an utterance, but exchange the speaker (or don't)

    Useful for Privacy Applications
    """

    def __init__(self, model_id, device, language="eng"):
        self.tts = ToucanTTSInterface(device=device, tts_model_path=model_id)
        self.ap = AudioPreprocessor(input_sr=100, output_sr=16000, cut_silence=False)
        self.tf = ArticulatoryCombinedTextFrontend(language=language, device=device)
        self.device = device
        acoustic_checkpoint_path = os.path.join(MODELS_DIR, "Aligner", "aligner.pt")
        self.aligner_weights = torch.load(acoustic_checkpoint_path, map_location=device)["asr_model"]
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True  # torch 1.9 has a bug in the hub loading, this is a workaround
        # careful: assumes 16kHz or 8kHz audio
        self.silero_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                  model='silero_vad',
                                                  force_reload=False,
                                                  onnx=False,
                                                  verbose=False)
        (self.get_speech_timestamps, _, _, _, _) = utils
        torch.set_grad_enabled(True)  # finding this issue was very infuriating: silero sets
        # this to false globally during model loading rather than using inference_mode or no_grad
        self.acoustic_model = Aligner()
        self.acoustic_model = self.acoustic_model.to(self.device)
        self.acoustic_model.load_state_dict(self.aligner_weights)
        self.acoustic_model.eval()
        self.parsel = Parselmouth(reduction_factor=1, fs=16000)
        self.energy_calc = EnergyCalculator(reduction_factor=1, fs=16000)
        self.dc = DurationCalculator(reduction_factor=1)

    def extract_prosody(self, transcript, ref_audio_path, lang="eng", on_line_fine_tune=True):
        if on_line_fine_tune:
            self.acoustic_model.load_state_dict(self.aligner_weights)
            self.acoustic_model.eval()

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

        with torch.inference_mode():
            speech_timestamps = self.get_speech_timestamps(norm_wave, self.silero_model, sampling_rate=16000)
        if len(speech_timestamps) == 0:
            speech_timestamps = [{'start': 0, 'end': len(norm_wave)}]
        start_silence = speech_timestamps[0]['start']
        end_silence = len(norm_wave) - speech_timestamps[-1]['end']
        norm_wave = norm_wave[speech_timestamps[0]['start']:speech_timestamps[-1]['end']]

        norm_wave_length = torch.LongTensor([len(norm_wave)])
        text = self.tf.string_to_tensor(transcript, handle_missing=False).squeeze(0)
        features = self.ap.audio_to_mel_spec_tensor(audio=norm_wave, explicit_sampling_rate=16000).transpose(0, 1)
        feature_length = torch.LongTensor([len(features)]).numpy()

        if on_line_fine_tune:
            # we fine-tune the aligner for a couple steps using SGD. This makes cloning pretty slow, but the results are greatly improved.
            steps = 4
            tokens = self.tf.text_vectors_to_id_sequence(text_vector=text)  # we need an ID sequence for training rather than a sequence of phonological features
            tokens = torch.LongTensor(tokens).squeeze().to(self.device)
            tokens_len = torch.LongTensor([len(tokens)]).to(self.device)
            mel = features.unsqueeze(0).to(self.device)
            mel_len = torch.LongTensor([len(mel[0])]).to(self.device)
            # actual fine-tuning starts here
            optim_asr = torch.optim.Adam(self.acoustic_model.parameters(), lr=0.00001)
            self.acoustic_model.train()
            for _ in range(steps):
                pred = self.acoustic_model(mel.clone())
                loss = self.acoustic_model.ctc_loss(pred.transpose(0, 1).log_softmax(2), tokens, mel_len, tokens_len)
                print(loss.item())
                optim_asr.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.acoustic_model.parameters(), 1.0)
                optim_asr.step()
            self.acoustic_model.eval()

        # We deal with the word boundaries by having 2 versions of text: with and without word boundaries.
        # We note the index of word boundaries and insert durations of 0 afterwards
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
        return duration, pitch, energy, start_silence, end_silence

    def clone_utterance(self,
                        path_to_reference_audio_for_intonation,
                        path_to_reference_audio_for_voice,
                        transcription_of_intonation_reference,
                        filename_of_result=None,
                        lang="eng"):
        """
        What is said in path_to_reference_audio_for_intonation has to match the text in the reference_transcription exactly!
        """
        self.tts.set_utterance_embedding(path_to_reference_audio=path_to_reference_audio_for_voice)
        duration, pitch, energy, silence_frames_start, silence_frames_end = self.extract_prosody(transcription_of_intonation_reference,
                                                                                                 path_to_reference_audio_for_intonation,
                                                                                                 lang=lang)
        self.tts.set_language(lang)
        start_sil = numpy.zeros([int(silence_frames_start * 1.5)])  # timestamps are from 16kHz, but now we're using 24000Hz, so upsampling required
        end_sil = numpy.zeros([int(silence_frames_end * 1.5)])  # timestamps are from 16kHz, but now we're using 24000Hz, so upsampling required
        cloned_speech, sr = self.tts(transcription_of_intonation_reference, view=False, durations=duration, pitch=pitch, energy=energy)
        cloned_utt = numpy.concatenate([start_sil, cloned_speech, end_sil], axis=0)
        if filename_of_result is not None:
            sf.write(file=filename_of_result, data=float2pcm(cloned_utt), samplerate=sr, subtype="PCM_16")

        return cloned_utt, sr
