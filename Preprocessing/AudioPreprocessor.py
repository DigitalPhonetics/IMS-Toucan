import librosa
import librosa.core as lb
import librosa.display as lbd
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pyloudnorm as pyln
import torch
from torchaudio.transforms import Resample


class AudioPreprocessor:

    def __init__(self, input_sr, output_sr=None, melspec_buckets=80, hop_length=256, n_fft=1024, cut_silence=False, device="cpu"):
        """
        The parameters are by default set up to do well
        on a 16kHz signal. A different sampling rate may
        require different hop_length and n_fft (e.g.
        doubling frequency --> doubling hop_length and
        doubling n_fft)
        """
        if device == "cpu" and cut_silence:
            print("Running silence removal on CPU is very slow, consider using GPU for this.")
        self.cut_silence = cut_silence
        self.device = device
        self.sr = input_sr
        self.new_sr = output_sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.mel_buckets = melspec_buckets
        self.meter = pyln.Meter(input_sr)
        self.final_sr = input_sr
        if cut_silence:
            torch.hub._validate_not_a_forked_repo = lambda a, b, c: True  # torch 1.9 has a bug in the hub loading, this is a workaround
            # careful: assumes 16kHz or 8kHz audio
            self.silero_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                      model='silero_vad',
                                                      force_reload=True,
                                                      onnx=False)
            (self.get_speech_timestamps,
             self.save_audio,
             self.read_audio,
             self.VADIterator,
             self.collect_chunks) = utils
            self.silero_model = self.silero_model.to(self.device)
        if output_sr is not None and output_sr != input_sr:
            self.resample = Resample(orig_freq=input_sr, new_freq=output_sr).to(self.device)
            self.final_sr = output_sr
        else:
            self.resample = lambda x: x

    def cut_silence_from_audio(self, audio):
        """
        https://github.com/snakers4/silero-vad
        """
        return self.collect_chunks(self.get_speech_timestamps(audio, self.silero_model, sampling_rate=self.final_sr), audio)

    def to_mono(self, x):
        """
        make sure we deal with a 1D array
        """
        if len(x.shape) == 2:
            return lb.to_mono(numpy.transpose(x))
        else:
            return x

    def normalize_loudness(self, audio):
        """
        normalize the amplitudes according to
        their decibels, so this should turn any
        signal with different magnitudes into
        the same magnitude by analysing loudness
        """
        loudness = self.meter.integrated_loudness(audio)
        loud_normed = pyln.normalize.loudness(audio, loudness, -30.0)
        peak = numpy.amax(numpy.abs(loud_normed))
        peak_normed = numpy.divide(loud_normed, peak)
        return peak_normed

    def logmelfilterbank(self, audio, sampling_rate, fmin=40, fmax=8000, eps=1e-10):
        """
        Compute log-Mel filterbank
        """
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        # get amplitude spectrogram
        x_stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=None, window="hann", pad_mode="reflect")
        spc = np.abs(x_stft).T
        # get mel basis
        fmin = 0 if fmin is None else fmin
        fmax = sampling_rate / 2 if fmax is None else fmax
        mel_basis = librosa.filters.mel(sampling_rate, self.n_fft, self.mel_buckets, fmin, fmax)
        # apply log and return
        return torch.Tensor(np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))).transpose(0, 1)

    def normalize_audio(self, audio):
        """
        one function to apply them all in an
        order that makes sense.
        """
        audio = self.to_mono(audio)
        audio = self.normalize_loudness(audio)
        audio = torch.Tensor(audio, device=self.device)
        audio = self.resample(audio)
        if self.cut_silence:
            audio = self.cut_silence_from_audio(audio)
        return audio

    def visualize_cleaning(self, unclean_audio):
        """
        displays Mel Spectrogram of unclean audio
        and then displays Mel Spectrogram of the
        cleaned version.
        """
        fig, ax = plt.subplots(nrows=2, ncols=1)
        unclean_audio_mono = self.to_mono(unclean_audio)
        unclean_spec = self.audio_to_mel_spec_tensor(unclean_audio_mono, normalize=False).numpy()
        clean_spec = self.audio_to_mel_spec_tensor(unclean_audio_mono, normalize=True).numpy()
        lbd.specshow(unclean_spec, sr=self.sr, cmap='GnBu', y_axis='mel', ax=ax[0], x_axis='time')
        ax[0].set(title='Uncleaned Audio')
        ax[0].label_outer()
        if self.new_sr is not None:
            lbd.specshow(clean_spec, sr=self.new_sr, cmap='GnBu', y_axis='mel', ax=ax[1], x_axis='time')
        else:
            lbd.specshow(clean_spec, sr=self.sr, cmap='GnBu', y_axis='mel', ax=ax[1], x_axis='time')
        ax[1].set(title='Cleaned Audio')
        ax[1].label_outer()
        plt.show()

    def audio_to_wave_tensor(self, audio, normalize=True):
        if normalize:
            return self.normalize_audio(audio)
        else:
            if isinstance(audio, torch.Tensor):
                return audio
            else:
                return torch.Tensor(audio)

    def audio_to_mel_spec_tensor(self, audio, normalize=True, explicit_sampling_rate=None):
        """
        explicit_sampling_rate is for when
        normalization has already been applied
        and that included resampling. No way
        to detect the current sr of the incoming
        audio
        """
        if explicit_sampling_rate is None:
            if normalize:
                audio = self.normalize_audio(audio)
                return self.logmelfilterbank(audio=audio, sampling_rate=self.final_sr)
            return self.logmelfilterbank(audio=audio, sampling_rate=self.sr)
        if normalize:
            audio = self.normalize_audio(audio)
        return self.logmelfilterbank(audio=audio, sampling_rate=explicit_sampling_rate)
