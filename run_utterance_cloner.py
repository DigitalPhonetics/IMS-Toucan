import os

import soundfile as sf
import torch
from numpy import trim_zeros

from InferenceInterfaces.Karlsson_FastSpeech2 import Karlsson_FastSpeech2
from InferenceInterfaces.Nancy_FastSpeech2 import Nancy_FastSpeech2
from Preprocessing.ArticulatoryCombinedTextFrontend import ArticulatoryCombinedTextFrontend
from Preprocessing.AudioPreprocessor import AudioPreprocessor
from TrainingInterfaces.Text_to_Spectrogram.AutoAligner.Aligner import Aligner
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.DurationCalculator import DurationCalculator
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.EnergyCalculator import EnergyCalculator
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.PitchCalculator import Dio

tts_dict = {
    "fast_nancy": Nancy_FastSpeech2,
    "fast_karlsson": Karlsson_FastSpeech2,
}


def extract_prosody(transcript, ref_audio_path, lang="de"):
    device = 'cpu'
    acoustic_model = Aligner()
    acoustic_checkpoint_path = os.path.join("Models", "Aligner", "aligner.pt")
    acoustic_model.load_state_dict(torch.load(acoustic_checkpoint_path, map_location='cpu')["asr_model"])
    acoustic_model = acoustic_model.to(device)
    dio = Dio(reduction_factor=1, fs=16000)
    energy_calc = EnergyCalculator(reduction_factor=1, fs=16000)
    dc = DurationCalculator(reduction_factor=1)
    wave, sr = sf.read(ref_audio_path)
    tf = ArticulatoryCombinedTextFrontend(language=lang, use_word_boundaries=False)
    ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024)
    try:
        norm_wave = ap.audio_to_wave_tensor(normalize=True, audio=wave)
    except ValueError:
        print('Something went wrong, the reference wave might be too short.')
    norm_wave = torch.tensor(trim_zeros(norm_wave.numpy()))
    norm_wave_length = torch.LongTensor([len(norm_wave)])
    text = tf.string_to_tensor(transcript, handle_missing=False).squeeze(0)
    melspec = ap.audio_to_mel_spec_tensor(audio=norm_wave, normalize=False, explicit_sampling_rate=16000).transpose(0, 1)
    melspec_length = torch.LongTensor([len(melspec)]).numpy()
    alignment_path = acoustic_model.inference(mel=melspec.to(device),
                                              tokens=text.to(device),
                                              return_ctc=False)
    duration = dc(torch.LongTensor(alignment_path), vis=None).cpu()
    energy = energy_calc(input_waves=norm_wave.unsqueeze(0),
                         input_waves_lengths=norm_wave_length,
                         feats_lengths=melspec_length,
                         durations=duration.unsqueeze(0),
                         durations_lengths=torch.LongTensor([len(duration)]))[0].squeeze(0).cpu()
    pitch = dio(input_waves=norm_wave.unsqueeze(0),
                input_waves_lengths=norm_wave_length,
                feats_lengths=melspec_length,
                durations=duration.unsqueeze(0),
                durations_lengths=torch.LongTensor([len(duration)]))[0].squeeze(0).cpu()
    # phones = tf.get_phone_string(transcript)
    # print(len(phones), " ", len(duration), " ", len(pitch), " ", len(energy))
    return transcript, duration, pitch, energy


def clone_utterance(path_to_reference_audio, reference_transcription, filename_of_result, model_id="fast_karlsson", device="cpu", lang="de"):
    tts = tts_dict[model_id](device=device)
    tts.set_utterance_embedding(path_to_reference_audio=path_to_reference_audio)
    transcript, duration, pitch, energy = extract_prosody(reference_transcription, path_to_reference_audio, lang=lang)
    tts.read_to_file(text_list=[reference_transcription], file_location=filename_of_result, dur_list=[duration], pitch_list=[pitch], energy_list=[energy])


if __name__ == '__main__':
    clone_utterance(path_to_reference_audio="audios/test.wav",
                    reference_transcription="Hello world, this is a test.",
                    filename_of_result="audios/test_cloned_cond_vctklibri.wav",
                    model_id="fast_nancy",
                    lang="en",
                    device="cpu")
