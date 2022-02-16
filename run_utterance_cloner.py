import os

import soundfile as sf
import torch
from numpy import trim_zeros
from torch.optim import SGD
from tqdm import tqdm

from InferenceInterfaces.Meta_FastSpeech2 import Meta_FastSpeech2
from InferenceInterfaces.MultiEnglish_FastSpeech2 import MultiEnglish_FastSpeech2
from InferenceInterfaces.MultiGerman_FastSpeech2 import MultiGerman_FastSpeech2
from Preprocessing.ArticulatoryCombinedTextFrontend import ArticulatoryCombinedTextFrontend
from Preprocessing.AudioPreprocessor import AudioPreprocessor
from TrainingInterfaces.Text_to_Spectrogram.AutoAligner.Aligner import Aligner
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.DurationCalculator import DurationCalculator
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.EnergyCalculator import EnergyCalculator
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.PitchCalculator import Dio

tts_dict = {
    "fast_meta": Meta_FastSpeech2,
    "fast_german": MultiGerman_FastSpeech2,
    "fast_english": MultiEnglish_FastSpeech2
}


def extract_prosody(transcript, ref_audio_path, lang="de", on_line_fine_tune=True):
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
    ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024, cut_silence=False)
    try:
        norm_wave = ap.audio_to_wave_tensor(normalize=True, audio=wave)
    except ValueError:
        print('Something went wrong, the reference wave might be too short.')
        raise RuntimeError
    norm_wave = torch.tensor(trim_zeros(norm_wave.numpy()))
    norm_wave_length = torch.LongTensor([len(norm_wave)])
    text = tf.string_to_tensor(transcript, handle_missing=False).squeeze(0)
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
        tokens = tokens.squeeze().to(device)
        tokens_len = torch.LongTensor([len(tokens)]).to(device)
        mel = melspec.squeeze().to(device)
        mel_len = torch.LongTensor([len(mel)]).to(device)
        # actual fine-tuning starts here
        optim_asr = SGD(acoustic_model.parameters(), lr=0.1)
        for _ in tqdm(list(range(steps))):
            acoustic_model.train()
            pred = acoustic_model(mel.unsqueeze(0), mel_len)
            loss = acoustic_model.ctc_loss(pred.transpose(0, 1).log_softmax(2), tokens, mel_len, tokens_len)
            optim_asr.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(acoustic_model.parameters(), 1.0)
            optim_asr.step()
        acoustic_model.eval()

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

    return duration, pitch, energy


def clone_utterance(path_to_reference_audio, reference_transcription, filename_of_result, clone_speaker_identity=True, model_id="fast_meta", device="cpu", lang="de"):
    tts = tts_dict[model_id](device=device)
    if clone_speaker_identity:
        tts.set_utterance_embedding(path_to_reference_audio=path_to_reference_audio)
    duration, pitch, energy = extract_prosody(reference_transcription, path_to_reference_audio, lang=lang)
    tts.set_language(lang)
    tts.read_to_file(text_list=[reference_transcription], file_location=filename_of_result, dur_list=[duration], pitch_list=[pitch], energy_list=[energy])


if __name__ == '__main__':
    clone_utterance(path_to_reference_audio="audios/test.wav",
                    reference_transcription="Hello world, this is a test.",
                    filename_of_result="audios/test_cloned.wav",
                    model_id="fast_meta",
                    lang="en",
                    device="cpu")
