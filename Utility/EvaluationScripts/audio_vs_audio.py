import librosa
import librosa.display as lbd
import matplotlib.pyplot as plt
import numpy
import soundfile as sf
import torch
from numpy import inf
from numpy import ndim
from numpy import zeros
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error

from Preprocessing.AudioPreprocessor import AudioPreprocessor
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.PitchCalculator_Dio import Dio
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.PitchCalculator import Parselmouth
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.PitchCalculator_Crepe import Crepe
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.PitchCalculator_Yin import Yin
from Utility.EvaluationScripts.soft_dtw import SoftDTW


def vde(path_1, path_2):
    """
    Voicing Decision Error measures the inverted accuracy of frames that are voiced compared to the reference.

    The first path should lead to the 'gold' audio
    """
    pitchcurve_1, pitchcurve_2 = get_pitch_curves(path_1, path_2)

    correct_frames, incorrect_frames = list(), list()
    for index in range(len(pitchcurve_1)):
        if (pitchcurve_1[index] == 0.0 and pitchcurve_2[index] != 0.0) or (pitchcurve_1[index] != 0.0 and pitchcurve_2[index] == 0.0):
            incorrect_frames.append(index)
        else:
            correct_frames.append(index)

    return len(incorrect_frames) / (len(correct_frames) + len(incorrect_frames))


def gpe(path_1, path_2):
    """
    Gross Pitch Error measures the percentage of voiced frames that deviate in pitch by more than 20% compared to the reference.

    The first path should lead to the 'gold' audio
    """
    pitchcurve_1, pitchcurve_2 = get_pitch_curves(path_1, path_2)

    correct_frames, incorrect_frames = list(), list()
    for index in range(len(pitchcurve_1)):
        twenty_percent_deviation = pitchcurve_1[index] * 0.2  # 20% deviation is acceptable
        if pitchcurve_1[index] + twenty_percent_deviation > pitchcurve_2[index] > pitchcurve_1[index] - twenty_percent_deviation:
            correct_frames.append(index)
        else:
            incorrect_frames.append(index)

    return len(incorrect_frames) / (len(correct_frames) + len(incorrect_frames))


def ffe(path_1, path_2):
    """
    F0 Frame Error measures the percentage of frames that either contain a 20% pitch error (according to GPE) or a voicing decision error (according to VDE).

    The first path should lead to the 'gold' audio
    """
    pitchcurve_1, pitchcurve_2 = get_pitch_curves(path_1, path_2)

    correct_frames, incorrect_frames = set(), set()
    for index in range(len(pitchcurve_1)):
        twenty_percent_deviation = pitchcurve_1[index] * 0.2  # 20% deviation is acceptable
        if (pitchcurve_1[index] + twenty_percent_deviation > pitchcurve_2[index] > pitchcurve_1[index] - twenty_percent_deviation) and not (
                (pitchcurve_1[index] == 0.0 and pitchcurve_2[index] != 0.0) or (pitchcurve_1[index] != 0.0 and pitchcurve_2[index] == 0.0)):
            correct_frames.add(index)
        else:
            incorrect_frames.add(index)

    return len(incorrect_frames) / (len(correct_frames) + len(incorrect_frames))


def mcd_with_warping(path_1, path_2):
    """
    calculate mel cepstral distortion between two unaligned sequences by performing alignment with warping using MSE as the distance between them.

    The two audios have to be spoken by the same speaker for it to make sense. The first one should be the gold reference.

    DTW takes an insane amount of RAM if you're not careful with sequence lengths
    """
    wave_1, sr_1 = sf.read(path_1)
    wave_2, sr_2 = sf.read(path_2)
    spec_1 = logmelfilterbank(audio=wave_1, sampling_rate=sr_1)
    spec_2 = logmelfilterbank(audio=wave_2, sampling_rate=sr_2)
    dist, _, _ = dtw(spec_1, spec_2, mean_squared_error)
    d = dist / len(spec_1)
    print(d)
    return d


@torch.inference_mode()
def soft_mcd(path_1, path_2):
    """
    calculate mel cepstral distortion between two unaligned sequences by performing alignment with warping using euclidean distance between them.

    The two audios have to be spoken by the same speaker for it to make sense. The first one should be the gold reference.
    """
    wave_1, sr_1 = sf.read(path_1)
    wave_2, sr_2 = sf.read(path_2)

    ap1 = AudioPreprocessor(cut_silence=True, input_sr=sr_1, output_sr=16000)
    ap2 = AudioPreprocessor(cut_silence=True, input_sr=sr_2, output_sr=16000)

    spec_1 = logmelfilterbank(audio=ap1.audio_to_wave_tensor(wave_1, normalize=True).squeeze().numpy(), sampling_rate=16000)
    spec_2 = logmelfilterbank(audio=ap2.audio_to_wave_tensor(wave_2, normalize=True).squeeze().numpy(), sampling_rate=16000)

    dist = SoftDTW(use_cuda=False, gamma=0.0001)(torch.tensor(spec_1).unsqueeze(0), torch.tensor(spec_2).unsqueeze(0)) / len(spec_2)
    print(dist)

    return dist


def dtw(x, y, dist, warp=1):
    """
    https://github.com/pierre-rouanet/dtw/blob/master/dtw/dtw.py
    """
    if ndim(x) == 1:
        x = x.reshape(-1, 1)
    if ndim(y) == 1:
        y = y.reshape(-1, 1)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]
    print("calculating alignment...")
    D0[1:, 1:] = cdist(x, y, dist)
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                min_list += [D0[min(i + k, r), j],
                             D0[i, min(j + k, c)]]
            D1[i, j] += min(min_list)
    return D1[-1, -1], C, D1


def logmelfilterbank(audio, sampling_rate, fmin=40, fmax=8000, eps=1e-10):
    # get amplitude spectrogram
    x_stft = librosa.stft(audio, n_fft=1024, hop_length=256, win_length=None, window="hann", pad_mode="reflect")
    spc = numpy.abs(x_stft).T
    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(sampling_rate, 1024, 80, fmin, fmax)
    # apply log and return
    return numpy.log10(numpy.maximum(eps, numpy.dot(spc, mel_basis.T)))


def get_pitch_curves(path_1, path_2, plot_curves=False, length_norm=True):
    wave_1, sr_1 = sf.read(path_1)
    wave_2, sr_2 = sf.read(path_2)

    ap_1 = AudioPreprocessor(cut_silence=True, input_sr=sr_1, output_sr=16000)
    ap_2 = AudioPreprocessor(cut_silence=True, input_sr=sr_2, output_sr=16000)

    norm_wave_1 = ap_1.audio_to_wave_tensor(wave_1, normalize=True)
    norm_wave_2 = ap_2.audio_to_wave_tensor(wave_2, normalize=True)

    dio = Dio(fs=16000, use_token_averaged_f0=False, use_log_f0=False, use_continuous_f0=False)

    pitch_curve_1 = dio(norm_wave_1.unsqueeze(0), norm_by_average=False)[0].squeeze()
    pitch_curve_2 = dio(norm_wave_2.unsqueeze(0), norm_by_average=False)[0].squeeze()

    if length_norm:
        # symmetrically remove samples from front and back, so we end up with the same amount of frames in both
        toggle = True
        while len(pitch_curve_1) > len(pitch_curve_2):
            if toggle:
                pitch_curve_1 = pitch_curve_1[1:]
            else:
                pitch_curve_1 = pitch_curve_1[:-1]
            toggle = not toggle
        while len(pitch_curve_1) < len(pitch_curve_2):
            if toggle:
                pitch_curve_2 = pitch_curve_2[1:]
            else:
                pitch_curve_2 = pitch_curve_2[:-1]
            toggle = not toggle

    if plot_curves:
        plt.plot(pitch_curve_1, c="red")
        plt.plot(pitch_curve_2, c="blue")
        plt.show()

    return pitch_curve_1, pitch_curve_2


def get_pitch_curves_abc(path_1, path_2, path_3):
    wave_1, sr_1 = sf.read(path_1)
    wave_2, sr_2 = sf.read(path_2)
    wave_3, sr_3 = sf.read(path_3)

    ap_1 = AudioPreprocessor(cut_silence=True, input_sr=sr_1, output_sr=16000, fmax_for_spec=1000)
    ap_2 = AudioPreprocessor(cut_silence=True, input_sr=sr_2, output_sr=16000, fmax_for_spec=1000)
    ap_3 = AudioPreprocessor(cut_silence=True, input_sr=sr_3, output_sr=16000, fmax_for_spec=1000)

    norm_wave_1 = ap_1.audio_to_wave_tensor(wave_1, normalize=True)
    norm_wave_2 = ap_2.audio_to_wave_tensor(wave_2, normalize=True)
    norm_wave_3 = ap_3.audio_to_wave_tensor(wave_3, normalize=True)

    dio = Dio(fs=16000, use_token_averaged_f0=False, use_log_f0=False, use_continuous_f0=False, n_fft=1024, hop_length=256)

    pitch_curve_1 = dio(norm_wave_1.unsqueeze(0), norm_by_average=False)[0].squeeze()
    pitch_curve_2 = dio(norm_wave_2.unsqueeze(0), norm_by_average=False)[0].squeeze()
    pitch_curve_3 = dio(norm_wave_3.unsqueeze(0), norm_by_average=False)[0].squeeze()

    fig, ax = plt.subplots(nrows=3, ncols=1)
    lbd.specshow(ap_1.audio_to_mel_spec_tensor(wave_1).numpy(),
                 ax=ax[0],
                 sr=16000,
                 cmap='GnBu',
                 y_axis='mel',
                 x_axis=None,
                 hop_length=256)
    ax[0].yaxis.set_visible(False)
    ax[0].set_title("Human Speech")
    ax[0].plot(pitch_curve_1, c="darkred")

    lbd.specshow(ap_2.audio_to_mel_spec_tensor(wave_2).numpy(),
                 ax=ax[1],
                 sr=16000,
                 cmap='GnBu',
                 y_axis='mel',
                 x_axis=None,
                 hop_length=256)
    ax[1].yaxis.set_visible(False)
    ax[1].set_title("Synthetic Speech 2")
    ax[1].plot(pitch_curve_2, c="darkred")

    lbd.specshow(ap_3.audio_to_mel_spec_tensor(wave_3).numpy(),
                 ax=ax[2],
                 sr=16000,
                 cmap='GnBu',
                 y_axis='mel',
                 x_axis=None,
                 hop_length=256)
    ax[2].yaxis.set_visible(False)
    ax[2].set_title("Synthetic Speech 1")
    ax[2].plot(pitch_curve_3, c="darkred")

    plt.tight_layout()
    plt.show()

def get_pitch_curve_diff_extractors(audio_path, text=None):
    wave, sr = sf.read(audio_path)

    ap = AudioPreprocessor(cut_silence=True, input_sr=sr, output_sr=16000, fmax_for_spec=1000)

    norm_wave = ap.audio_to_wave_tensor(wave, normalize=True)
    dio = Dio(fs=16000, use_token_averaged_f0=False, use_log_f0=False, use_continuous_f0=False, n_fft=1024, hop_length=256)
    parsel = Parselmouth(fs=16000, use_token_averaged_f0=False, use_log_f0=False, use_continuous_f0=False, n_fft=1024, hop_length=256)
    crepe = Crepe(fs=16000, use_token_averaged_f0=False, use_log_f0=False, use_continuous_f0=False, n_fft=1024, hop_length=256)
    yin = Yin(fs=16000, use_token_averaged_f0=False, use_log_f0=False, use_continuous_f0=False, n_fft=1024, hop_length=256)


    pitch_curve_1 = dio(norm_wave.unsqueeze(0), norm_by_average=False)[0].squeeze()
    pitch_curve_2 = parsel(norm_wave.unsqueeze(0), norm_by_average=False)[0].squeeze()
    pitch_curve_3 = crepe(norm_wave.unsqueeze(0), norm_by_average=False)[0].squeeze()
    pitch_curve_4 = yin(norm_wave.unsqueeze(0), norm_by_average=False)[0].squeeze()

    print(norm_wave.shape)
    print('Dio\n', pitch_curve_1, "\n", len(pitch_curve_1))
    print('Parsel\n', pitch_curve_2, "\n", len(pitch_curve_2))
    print('Crepe\n', pitch_curve_3, "\n", len(pitch_curve_3))
    print('Yin\n', pitch_curve_3, "\n", len(pitch_curve_4))
    
    # plt.plot(pitch_curve_1, c="red")
    # plt.plot(pitch_curve_2, c="blue")
    # plt.plot(pitch_curve_3, c="green")
    # plt.show()

    fig, ax = plt.subplots(nrows=4, ncols=1)
    lbd.specshow(ap.audio_to_mel_spec_tensor(wave).numpy(),
                 ax=ax[0],
                 sr=16000,
                 cmap='GnBu',
                 y_axis='mel',
                 x_axis=None,
                 hop_length=256)
    ax[0].yaxis.set_visible(False)
    ax[0].set_title("Dio")
    ax[0].plot(pitch_curve_1, c="darkred")

    lbd.specshow(ap.audio_to_mel_spec_tensor(wave).numpy(),
                 ax=ax[1],
                 sr=16000,
                 cmap='GnBu',
                 y_axis='mel',
                 x_axis=None,
                 hop_length=256)
    ax[1].yaxis.set_visible(False)
    ax[1].set_title("Parselmouth")
    ax[1].plot(pitch_curve_2, c="darkred")


    
    lbd.specshow(ap.audio_to_mel_spec_tensor(wave).numpy(),
                 ax=ax[2],
                 sr=16000,
                 cmap='GnBu',
                 y_axis='mel',
                 x_axis=None,
                 hop_length=256)
    ax[2].yaxis.set_visible(False)
    ax[2].set_title("Crepe")
    ax[2].plot(pitch_curve_3, c="darkred")


    lbd.specshow(ap.audio_to_mel_spec_tensor(wave).numpy(),
                 ax=ax[3],
                 sr=16000,
                 cmap='GnBu',
                 y_axis='mel',
                 x_axis=None,
                 hop_length=256)
    ax[3].yaxis.set_visible(False)
    ax[3].set_title("Yin")
    ax[3].plot(pitch_curve_4, c="darkred")

    plt.tight_layout()
    plt.show()