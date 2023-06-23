from Preprocessing.AudioPreprocessor import AudioPreprocessor
import soundfile as sf
import torch
from numpy import trim_zeros
from InferenceInterfaces.InferenceArchitectures.InferenceBigVGAN import BigVGAN
from InferenceInterfaces.InferenceArchitectures.InferenceAvocodo import HiFiGANGenerator
import soundfile
from Utility.utils import float2pcm
import os
from Utility.storage_config import MODELS_DIR

if __name__ == '__main__':
    paths_female = ["/mount/resources/speech/corpora/Emotional_Speech_Dataset_Singapore/0015/Angry/0015_000605.wav",
                    "/mount/resources/speech/corpora/Emotional_Speech_Dataset_Singapore/0015/Happy/0015_000814.wav",
                    "/mount/resources/speech/corpora/Emotional_Speech_Dataset_Singapore/0015/Neutral/0015_000148.wav",
                    "/mount/resources/speech/corpora/Emotional_Speech_Dataset_Singapore/0015/Sad/0015_001088.wav",
                    "/mount/resources/speech/corpora/Emotional_Speech_Dataset_Singapore/0015/Surprise/0015_001604.wav"]
    ids_female = [0, 1, 1, 0, 0, 0, 1]
    emotions_female = ["anger", "joy", "neutral", "sadness", "surprise"]

    paths_female2 = ["/mount/resources/speech/corpora/RAVDESS/Actor_12/03-01-05-01-01-02-12.wav",
                     "/mount/resources/speech/corpora/RAVDESS/Actor_12/03-01-07-01-01-01-12.wav",
                     "/mount/resources/speech/corpora/RAVDESS/Actor_12/03-01-06-01-01-02-12.wav",
                     "/mount/resources/speech/corpora/RAVDESS/Actor_12/03-01-03-01-02-02-12.wav",
                     "/mount/resources/speech/corpora/RAVDESS/Actor_12/03-01-01-01-02-01-12.wav",
                     "/mount/resources/speech/corpora/RAVDESS/Actor_12/03-01-04-01-02-01-12.wav",
                     "/mount/resources/speech/corpora/RAVDESS/Actor_12/03-01-08-01-01-01-12.wav"]
    ids_female2 = [1, 0, 0, 1, 1, 1, 0]
    emotions_female2 = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

    vocoder_model_path = os.path.join(MODELS_DIR, "Avocodo", "best.pt")
    mel2wav = HiFiGANGenerator(path_to_weights=vocoder_model_path).to(torch.device('cpu'))
    mel2wav.remove_weight_norm()
    mel2wav.eval()

    for i, path in enumerate(paths_female2):
        wave, sr = sf.read(path)
        ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024, cut_silence=True, device='cpu')
        norm_wave = ap.audio_to_wave_tensor(normalize=True, audio=wave)
        norm_wave = torch.tensor(trim_zeros(norm_wave.numpy()))
        spec = ap.audio_to_mel_spec_tensor(audio=norm_wave, normalize=False, explicit_sampling_rate=16000).cpu()

        wave = mel2wav(spec)
        silence = torch.zeros([10600])
        wav = silence.clone()
        wav = torch.cat((wav, wave, silence), 0)

        wav = [val for val in wav.detach().numpy() for _ in (0, 1)]  # doubling the sampling rate for better compatibility (24kHz is not as standard as 48kHz)
        soundfile.write(file=f"./audios/Original/female2/orig_{emotions_female2[i]}_{ids_female2[i]}.flac", data=float2pcm(wav), samplerate=48000, subtype="PCM_16")