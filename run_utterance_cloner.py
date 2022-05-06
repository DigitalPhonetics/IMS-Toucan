import os

import soundfile as sf
import torch
from torch.optim import SGD
from tqdm import tqdm

from InferenceInterfaces.InferenceFastSpeech2 import InferenceFastSpeech2
from Preprocessing.ArticulatoryCombinedTextFrontend import ArticulatoryCombinedTextFrontend
from Preprocessing.AudioPreprocessor import AudioPreprocessor
from TrainingInterfaces.Text_to_Spectrogram.AutoAligner.Aligner import Aligner
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.DurationCalculator import DurationCalculator
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.EnergyCalculator import EnergyCalculator
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.PitchCalculator import Dio
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.PitchCalculator_Parselmouth import Parselmouth
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.PitchCalculator_Crepe import Crepe
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.PitchCalculator_Yin import Yin


class UtteranceCloner:

    def __init__(self, model_id, device):
        self.tts = InferenceFastSpeech2(device=device, model_name=model_id)
        self.device = device
        acoustic_checkpoint_path = os.path.join("Models", "Aligner", "aligner.pt")
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
        # this to false globally during model loading rather than using inference mode or no_grad

    def extract_prosody(self, transcript, ref_audio_path, lang="de", on_line_fine_tune=True):
        acoustic_model = Aligner()
        acoustic_model.load_state_dict(self.aligner_weights)
        acoustic_model = acoustic_model.to(self.device)
        # dio = Dio(reduction_factor=1, fs=16000)
        # parsel = Parselmouth(reduction_factor=1, fs=16000)
        # crepe = Crepe(reduction_factor=1, fs=16000)
        yin = Yin(reduction_factor=1, fs=16000)
        energy_calc = EnergyCalculator(reduction_factor=1, fs=16000)
        dc = DurationCalculator(reduction_factor=1)
        wave, sr = sf.read(ref_audio_path)
        tf = ArticulatoryCombinedTextFrontend(language=lang, use_word_boundaries=False)
        ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024, cut_silence=False)
        try:
            norm_wave = ap.audio_to_wave_tensor(normalize=True, audio=wave)
        except ValueError:
            print('Something went wrong, the reference wave might be too short.')
            raise RuntimeError

        with torch.inference_mode():
            speech_timestamps = self.get_speech_timestamps(norm_wave, self.silero_model, sampling_rate=16000)
        norm_wave = norm_wave[speech_timestamps[0]['start']:speech_timestamps[-1]['end']]

        norm_wave_length = torch.LongTensor([len(norm_wave)])
        text = tf.string_to_tensor(transcript, handle_missing=False).squeeze(0)
        print(text)
        melspec = ap.audio_to_mel_spec_tensor(audio=norm_wave, normalize=False, explicit_sampling_rate=16000).transpose(0, 1)
        melspec_length = torch.LongTensor([len(melspec)]).numpy()

        if on_line_fine_tune:
            # we fine-tune the aligner for a couple steps using SGD. This makes cloning pretty slow, but the results are greatly improved.
            steps = 10
            tokens = list()  # we need an ID sequence for training rather than a sequence of phonological features
            for vector in text:
                for phone in tf.phone_to_vector:
                    if vector.numpy().tolist() == tf.phone_to_vector[phone]:
                        tokens.append(tf.phone_to_id[phone])
            tokens = torch.LongTensor(tokens)
            tokens = tokens.squeeze().to(self.device)
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

        alignment_path = acoustic_model.inference(mel=melspec.to(self.device),
                                                  tokens=text.to(self.device),
                                                  return_ctc=False)

        duration = dc(torch.LongTensor(alignment_path), vis=None).cpu()
        energy = energy_calc(input_waves=norm_wave.unsqueeze(0),
                             input_waves_lengths=norm_wave_length,
                             feats_lengths=melspec_length,
                             durations=duration.unsqueeze(0),
                             durations_lengths=torch.LongTensor([len(duration)]))[0].squeeze(0).cpu()
        # pitch = dio(input_waves=norm_wave.unsqueeze(0),
        #             input_waves_lengths=norm_wave_length,
        #             feats_lengths=melspec_length,
        #             durations=duration.unsqueeze(0),
        #             durations_lengths=torch.LongTensor([len(duration)]))[0].squeeze(0).cpu()
        # pitch = parsel(input_waves=norm_wave.unsqueeze(0),
        #             input_waves_lengths=norm_wave_length,
        #             feats_lengths=melspec_length,
        #             durations=duration.unsqueeze(0),
        #             durations_lengths=torch.LongTensor([len(duration)]))[0].squeeze(0).cpu()
        # pitch = crepe(input_waves=norm_wave.unsqueeze(0),
        #             input_waves_lengths=norm_wave_length,
        #             feats_lengths=melspec_length,
        #             durations=duration.unsqueeze(0),
        #             durations_lengths=torch.LongTensor([len(duration)]))[0].squeeze(0).cpu()
        pitch = yin(input_waves=norm_wave.unsqueeze(0),
                    input_waves_lengths=norm_wave_length,
                    feats_lengths=melspec_length,
                    durations=duration.unsqueeze(0),
                    durations_lengths=torch.LongTensor([len(duration)]))[0].squeeze(0).cpu()

        return duration, pitch, energy, speech_timestamps[0]['start'], speech_timestamps[-1]['end']

    def clone_utterance(self,
                        path_to_reference_audio,
                        reference_transcription,
                        filename_of_result,
                        clone_speaker_identity=True,
                        lang="de"):
        if clone_speaker_identity:
            prev_speaker_embedding = self.tts.default_utterance_embedding.clone().detach()
            self.tts.set_utterance_embedding(path_to_reference_audio=path_to_reference_audio)
        duration, pitch, energy, silence_frames_start, silence_frames_end = self.extract_prosody(reference_transcription,
                                                                                                 path_to_reference_audio,
                                                                                                 lang=lang)
        self.tts.set_language(lang)
        start_sil = torch.zeros([silence_frames_start * 3]).to(self.device)  # timestamps are from 16kHz, but now we're using 48kHz, so upsampling required
        end_sil = torch.zeros([silence_frames_end * 3]).to(self.device)  # timestamps are from 16kHz, but now we're using 48kHz, so upsampling required
        cloned_speech = self.tts(reference_transcription, view=False, durations=duration, pitch=pitch, energy=energy)
        cloned_utt = torch.cat((start_sil, cloned_speech, end_sil), dim=0)
        sf.write(file=filename_of_result, data=cloned_utt.cpu().numpy(), samplerate=48000)
        if clone_speaker_identity:
            self.tts.default_utterance_embedding = prev_speaker_embedding.to(self.device)  # return to normal


if __name__ == '__main__':
    uc = UtteranceCloner(model_id="Meta", device="cuda" if torch.cuda.is_available() else "cpu")

    uc.clone_utterance(path_to_reference_audio="/Users/kockja/Documents/PhD/adept/human/3.wav",
                       reference_transcription="Flour is in a separate aisle from other baking supplies?",
                       filename_of_result="audios/test_yin_3.wav",
                       clone_speaker_identity=True,
                       lang="en")

    # uc.clone_utterance(path_to_reference_audio="audios/test.wav",
    #                    reference_transcription="Hello, world, this, is, a, test.",
    #                    filename_of_result="audios/test_cloned_unnecessary_pauses.wav",
    #                    clone_speaker_identity=False,
    #                    lang="en")

    # uc.clone_utterance(path_to_reference_audio="audios/test_sing.wav",
    #                    reference_transcription="It was, one hundred degrees, as we sat, beneath, a willow tree, whose tears didn't care, they just hung in the air and refused, to fall.",
    #                    filename_of_result="audios/test_sing_cloned.wav",
    #                    clone_speaker_identity=False,
    #                    lang="en")
