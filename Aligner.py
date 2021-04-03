import json
import os

import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from FastSpeech2.DurationCalculator import DurationCalculator
from PreprocessingForTTS.ProcessAudio import AudioPreprocessor
from PreprocessingForTTS.ProcessText import TextFrontend
from TransformerTTS.TransformerTTS import build_reference_transformer_tts_model


def align(path_to_transcript_dict,
          acoustic_model_name,
          spemb,
          cache_dir,
          lang,
          reduction_factor=1,
          device=torch.device("cpu")):
    spemb = spemb
    transcript_to_durations = dict()
    path_list = list(path_to_transcript_dict.keys())
    tf = TextFrontend(language=lang,
                      use_panphon_vectors=False,
                      use_word_boundaries=False,
                      use_explicit_eos=False)
    _, sr = sf.read(path_list[0])
    if os.path.isdir(os.path.join(cache_dir, "alignments_visualization")):
        # reset duration sanity check dir
        os.removedirs(os.path.join(cache_dir, "alignments_visualization"))
    os.makedirs(os.path.join(cache_dir, "alignments_visualization"))
    if spemb:
        wav2mel = torch.jit.load("Models/Use/SpeakerEmbedding/wav2mel.pt")
        dvector = torch.jit.load("Models/Use/SpeakerEmbedding/dvector-step250000.pt").eval()
    ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024)
    acoustic_model = build_reference_transformer_tts_model(model_name=acoustic_model_name).to(device)
    dc = DurationCalculator(reduction_factor=reduction_factor)
    for index, path in tqdm(enumerate(path_list)):
        transcript = path_to_transcript_dict[path]
        wave, sr = sf.read(path)
        norm_wave = ap.audio_to_wave_tensor(audio=wave, normalize=True, mulaw=False)
        melspec = ap.audio_to_mel_spec_tensor(norm_wave, normalize=False).transpose(0, 1)
        text = tf.string_to_tensor(transcript).long()
        cached_text = tf.string_to_tensor(transcript).squeeze(0).numpy().tolist()
        if not spemb:
            cached_durations = dc(acoustic_model.inference(text=text.squeeze(0).to(device),
                                                           speech=melspec.to(device),
                                                           use_teacher_forcing=True,
                                                           spembs=None)[2],
                                  vis=os.path.join(cache_dir, "alignments_visualization",
                                                   path.split("/")[-1].rstrip(".wav") + ".png"))[
                0].cpu().numpy().tolist()
        else:
            wav_tensor, sample_rate = torchaudio.load(path)
            mel_tensor = wav2mel(wav_tensor, sample_rate)
            cached_spemb = dvector.embed_utterance(mel_tensor)
            cached_durations = dc(acoustic_model.inference(text=text.squeeze(0).to(device),
                                                           speech=melspec.to(device),
                                                           use_teacher_forcing=True,
                                                           spembs=cached_spemb.to(device))[2])[0].cpu().numpy().tolist()

        durations_in_seconds = list()
        for duration in cached_durations:
            durations_in_seconds.append((float(duration) / len(melspec)) * (len(wave) / sr))

        transcript_to_durations[path] = [cached_text, durations_in_seconds]

    with open(os.path.join(cache_dir, "alignments.json"), 'w') as fp:
        json.dump(transcript_to_durations, fp)


if __name__ == '__main__':
    from Utility.path_to_transcript_dicts import build_path_to_transcript_dict_ljspeech

    align(path_to_transcript_dict=build_path_to_transcript_dict_ljspeech(),
          acoustic_model_name="Transformer_English_Single.pt",
          spemb=False,
          cache_dir="Corpora/LJSpeech",
          lang="en",
          reduction_factor=1,
          device=torch.device("cpu"))
