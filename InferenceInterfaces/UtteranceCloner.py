import os

import soundfile as sf
import torch
from torch.optim import SGD
from tqdm import tqdm

from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface
from Preprocessing.AudioPreprocessor import AudioPreprocessor
from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
from Preprocessing.articulatory_features import get_feature_to_index_lookup
from TrainingInterfaces.Text_to_Spectrogram.AutoAligner.Aligner import Aligner
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.DurationCalculator import DurationCalculator
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.EnergyCalculator import EnergyCalculator
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.PitchCalculator import Parselmouth
from Utility.storage_config import MODELS_DIR


class UtteranceCloner:
    """
    Clone the prosody of an utterance, but exchange the speaker (or don't)

    Useful for Privacy Applications
    """

    def __init__(self, model_id, device, language="en", speed_over_quality=False):
        if (device == torch.device("cpu") or device == "cpu") and not speed_over_quality:
            print("Warning: You are running BigVGAN on CPU. Consider either switching to GPU or setting the speed_over_quality option to True.")
        self.tts = ToucanTTSInterface(device=device, tts_model_path=model_id, faster_vocoder=speed_over_quality)
        self.ap = AudioPreprocessor(input_sr=16000, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024, cut_silence=False)
        self.tf = ArticulatoryCombinedTextFrontend(language=language)
        self.device = device
        acoustic_checkpoint_path = os.path.join(MODELS_DIR, "Aligner", "aligner.pt")
        self.aligner_weights = torch.load(acoustic_checkpoint_path, map_location='cpu')["asr_model"]
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

    def extract_prosody(self, transcript, ref_audio_path, lang="de", on_line_fine_tune=True):
        acoustic_model = Aligner()
        acoustic_model.load_state_dict(self.aligner_weights)
        acoustic_model = acoustic_model.to(self.device)
        parsel = Parselmouth(reduction_factor=1, fs=16000)
        energy_calc = EnergyCalculator(reduction_factor=1, fs=16000)
        dc = DurationCalculator(reduction_factor=1)
        wave, sr = sf.read(ref_audio_path)
        if self.tf.language != lang:
            self.tf = ArticulatoryCombinedTextFrontend(language=lang)
        if self.ap.sr != sr:
            self.ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024, cut_silence=False)
        try:
            norm_wave = self.ap.audio_to_wave_tensor(normalize=True, audio=wave)
        except ValueError:
            print('Something went wrong, the reference wave might be too short.')
            raise RuntimeError

        with torch.inference_mode():
            speech_timestamps = self.get_speech_timestamps(norm_wave, self.silero_model, sampling_rate=16000)
        start_silence = speech_timestamps[0]['start']
        end_silence = len(norm_wave) - speech_timestamps[-1]['end']
        norm_wave = norm_wave[speech_timestamps[0]['start']:speech_timestamps[-1]['end']]

        norm_wave_length = torch.LongTensor([len(norm_wave)])
        text = self.tf.string_to_tensor(transcript, handle_missing=False).squeeze(0)
        melspec = self.ap.audio_to_mel_spec_tensor(audio=norm_wave, normalize=False, explicit_sampling_rate=16000).transpose(0, 1)
        melspec_length = torch.LongTensor([len(melspec)]).numpy()

        if on_line_fine_tune:
            # we fine-tune the aligner for a couple steps using SGD. This makes cloning pretty slow, but the results are greatly improved.
            steps = 5
            tokens = self.tf.text_vectors_to_id_sequence(text_vector=text)  # we need an ID sequence for training rather than a sequence of phonological features
            tokens = torch.LongTensor(tokens).squeeze().to(self.device)
            tokens_len = torch.LongTensor([len(tokens)]).to(self.device)
            mel = melspec.unsqueeze(0).to(self.device)
            mel.requires_grad = True
            mel_len = torch.LongTensor([len(mel[0])]).to(self.device)
            # actual fine-tuning starts here
            optim_asr = SGD(acoustic_model.parameters(), lr=0.1)
            acoustic_model.train()
            for _ in tqdm(list(range(steps))):
                pred = acoustic_model(mel)
                loss = acoustic_model.ctc_loss(pred.transpose(0, 1).log_softmax(2), tokens, mel_len, tokens_len)
                optim_asr.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(acoustic_model.parameters(), 1.0)
                optim_asr.step()
            acoustic_model.eval()

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

        alignment_path = acoustic_model.inference(mel=melspec.to(self.device),
                                                  tokens=matrix_without_word_boundaries.to(self.device),
                                                  return_ctc=False)

        duration = dc(torch.LongTensor(alignment_path), vis=None).cpu()

        for index_of_word_boundary in indexes_of_word_boundaries:
            duration = torch.cat([duration[:index_of_word_boundary],
                                  torch.LongTensor([0]),  # insert a 0 duration wherever there is a word boundary
                                  duration[index_of_word_boundary:]])

        last_vec = None
        for phoneme_index, vec in enumerate(text):
            if last_vec is not None:
                if last_vec.numpy().tolist() == vec.numpy().tolist():
                    # we found a case of repeating phonemes!
                    # now we must repair their durations by giving the first one 3/5 of their sum and the second one 2/5 (i.e. the rest)
                    dur_1 = duration[phoneme_index - 1]
                    dur_2 = duration[phoneme_index]
                    total_dur = dur_1 + dur_2
                    new_dur_1 = int((total_dur / 5) * 3)
                    new_dur_2 = total_dur - new_dur_1
                    duration[phoneme_index - 1] = new_dur_1
                    duration[phoneme_index] = new_dur_2
            last_vec = vec

        energy = energy_calc(input_waves=norm_wave.unsqueeze(0),
                             input_waves_lengths=norm_wave_length,
                             feats_lengths=melspec_length,
                             text=text,
                             durations=duration.unsqueeze(0),
                             durations_lengths=torch.LongTensor([len(duration)]))[0].squeeze(0).cpu()
        pitch = parsel(input_waves=norm_wave.unsqueeze(0),
                       input_waves_lengths=norm_wave_length,
                       feats_lengths=melspec_length,
                       text=text,
                       durations=duration.unsqueeze(0),
                       durations_lengths=torch.LongTensor([len(duration)]))[0].squeeze(0).cpu()
        return duration, pitch, energy, start_silence, end_silence

    def clone_utterance(self,
                        path_to_reference_audio_for_intonation,
                        path_to_reference_audio_for_voice,
                        transcription_of_intonation_reference,
                        filename_of_result=None,
                        lang="de"):
        """
        What is said in path_to_reference_audio_for_intonation has to match the text in the reference_transcription exactly!
        """
        self.tts.set_utterance_embedding(path_to_reference_audio=path_to_reference_audio_for_voice)
        duration, pitch, energy, silence_frames_start, silence_frames_end = self.extract_prosody(transcription_of_intonation_reference,
                                                                                                 path_to_reference_audio_for_intonation,
                                                                                                 lang=lang)
        self.tts.set_language(lang)
        start_sil = torch.zeros([silence_frames_start * 3]).to(self.device)  # timestamps are from 16kHz, but now we're using 48kHz, so upsampling required
        end_sil = torch.zeros([silence_frames_end * 3]).to(self.device)  # timestamps are from 16kHz, but now we're using 48kHz, so upsampling required
        cloned_speech = self.tts(transcription_of_intonation_reference, view=False, durations=duration, pitch=pitch, energy=energy)
        cloned_utt = torch.cat((start_sil, cloned_speech, end_sil), dim=0).cpu().numpy()
        if filename_of_result is not None:
            sf.write(file=filename_of_result, data=cloned_utt, samplerate=24000)
        return cloned_utt

    def biblical_accurate_angel_mode(self,
                                     path_to_reference_audio_for_intonation,
                                     transcription_of_intonation_reference,
                                     list_of_speaker_references_for_ensemble,
                                     filename_of_result=None,
                                     lang="de"):
        """
        Have multiple voices speak with the exact same intonation simultaneously
        """
        prev_embedding = self.tts.default_utterance_embedding.clone().detach()
        duration, pitch, energy, silence_frames_start, silence_frames_end = self.extract_prosody(transcription_of_intonation_reference,
                                                                                                 path_to_reference_audio_for_intonation,
                                                                                                 lang=lang)
        self.tts.set_language(lang)
        start_sil = torch.zeros([silence_frames_start * 3]).to(self.device)  # timestamps are from 16kHz, but now we're using 48kHz, so upsampling required
        end_sil = torch.zeros([silence_frames_end * 3]).to(self.device)  # timestamps are from 16kHz, but now we're using 48kHz, so upsampling required
        list_of_cloned_speeches = list()
        for path in list_of_speaker_references_for_ensemble:
            self.tts.set_utterance_embedding(path_to_reference_audio=path)
            list_of_cloned_speeches.append(self.tts(transcription_of_intonation_reference, view=False, durations=duration, pitch=pitch, energy=energy))
        cloned_speech = torch.stack(list_of_cloned_speeches).mean(dim=0)
        cloned_utt = torch.cat((start_sil, cloned_speech, end_sil), dim=0).cpu().numpy()
        if filename_of_result is not None:
            sf.write(file=filename_of_result, data=cloned_utt, samplerate=24000)
        self.tts.default_utterance_embedding = prev_embedding.to(self.device)  # return to normal
        return cloned_utt
