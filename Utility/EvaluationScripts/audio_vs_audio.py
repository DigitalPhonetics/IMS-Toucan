"""
gross pitch error
Where pt,p
0
t
are the pitch signals from the reference
and predicted audio, vt,v
0
t
are the voicing decisions
from the reference and predicted audio, and 1 is the
indicator function. The GPE measures the percentage
of voiced frames that deviate in pitch by more than
20% compared to the reference.




voicing decision error
Where vt,v
0
t
are the voicing decisions for the reference
and predicted audio, T is the total number of frames,
and 1 is the indicator function.




f0 frame error
FFE measures the percentage of frames that either contain a 20% pitch error (according to GPE) or a voicing decision error (according to VDE).

"""

import librosa
import numpy
import soundfile as sf
from numpy import inf
from numpy import ndim
from numpy import zeros
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error


def mcd_with_warping(path_1, path_2):
    """
    calculate mel cepstral distortion between two unaligned sequences by first performing alignment with warping and then calculating the MSE between them.

    DTW takes an insane amount of RAM if you're not careful with sequence lengths
    """
    wave_1, sr_1 = sf.read(path_1)
    wave_2, sr_2 = sf.read(path_2)
    spec_1 = logmelfilterbank(audio=wave_1, sampling_rate=sr_1)
    spec_2 = logmelfilterbank(audio=wave_2, sampling_rate=sr_2)
    dist, _, _ = dtw(spec_1, spec_2, mean_squared_error)
    return dist


def dtw(x, y, dist, warp=1):
    """
        https://github.com/pierre-rouanet/dtw/blob/master/dtw/dtw.py
    """
    assert len(x)
    assert len(y)
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
