import librosa
import librosa.core as lb
import librosa.display as lbd
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pyloudnorm as pyln
import torch
from torchaudio.transforms import Resample


def to_mono(x):
    """
    make sure we deal with a 1D array
    """
    if len(x.shape) == 2:
        return lb.to_mono(numpy.transpose(x))
    else:
        return x


class AudioPreprocessor:

    def __init__(self, input_sr, output_sr=None, melspec_buckets=80, hop_length=256, n_fft=1024, cut_silence=False, device="cpu", fmax_for_spec=8000):
        """
        The parameters are by default set up to do well
        on a 16kHz signal. A different sampling rate may
        require different hop_length and n_fft (e.g.
        doubling frequency --> doubling hop_length and
        doubling n_fft)
        """
        self.cut_silence = cut_silence
        self.device = device
        self.sr = input_sr
        self.new_sr = output_sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.mel_buckets = melspec_buckets
        self.meter = pyln.Meter(input_sr)
        self.final_sr = input_sr
        self.fmax_for_spec = fmax_for_spec
        if cut_silence:
            torch.hub._validate_not_a_forked_repo = lambda a, b, c: True  # torch 1.9 has a bug in the hub loading, this is a workaround
            # careful: assumes 16kHz or 8kHz audio
            self.silero_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                      model='silero_vad',
                                                      force_reload=False,
                                                      onnx=False,
                                                      verbose=False)
            (self.get_speech_timestamps,
             self.save_audio,
             self.read_audio,
             self.VADIterator,
             self.collect_chunks) = utils
            torch.set_grad_enabled(True)  # finding this issue was very infuriating: silero sets
            # this to false globally during model loading rather than using inference mode or no_grad
            self.silero_model = self.silero_model.to(self.device)
        else:
            self.device = "cpu"  # if we don't run the VAD model, there's simply no reason to use the GPU.
        if output_sr is not None and output_sr != input_sr:
            self.resample = Resample(orig_freq=input_sr, new_freq=output_sr).to(self.device)
            self.final_sr = output_sr
        else:
            self.resample = lambda x: x

    def cut_silence_from_audio(self, audio):
        """
        https://github.com/snakers4/silero-vad
        """
        with torch.inference_mode():
            speech_timestamps = self.get_speech_timestamps(audio, self.silero_model, sampling_rate=self.final_sr)
        try:
            result = audio[speech_timestamps[0]['start']:speech_timestamps[-1]['end']]
            return result
        except IndexError:
            print("Audio might be too short to cut silences from front and back.")
        return audio

    def normalize_loudness(self, audio):
        """
        normalize the amplitudes according to
        their decibels, so this should turn any
        signal with different magnitudes into
        the same magnitude by analysing loudness
        """
        try:
            loudness = self.meter.integrated_loudness(audio)
        except ValueError:
            # if the audio is too short, a value error will arise
            return audio
        loud_normed = pyln.normalize.loudness(audio, loudness, -30.0)
        peak = numpy.amax(numpy.abs(loud_normed))
        peak_normed = numpy.divide(loud_normed, peak)
        return peak_normed

    def logmelfilterbank(self, audio, sampling_rate, fmin=40, fmax=None, eps=1e-10):
        """
        Compute log-Mel filterbank

        one day this could be replaced by torchaudio's internal log10(melspec(audio)), but
        for some reason it gives slightly different results, so in order not to break backwards
        compatibility, this is kept for now. If there is ever a reason to completely re-train
        all models, this would be a good opportunity to make the switch.
        """
        if fmax is None:
            fmax = self.fmax_for_spec
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        # get amplitude spectrogram
        x_stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=None, window="hann", pad_mode="reflect")
        spc = np.abs(x_stft).T
        # get mel basis
        fmin = 0 if fmin is None else fmin
        fmax = sampling_rate / 2 if fmax is None else fmax
        mel_basis = librosa.filters.mel(sr=sampling_rate, n_fft=self.n_fft, n_mels=self.mel_buckets, fmin=fmin, fmax=fmax)
        # apply log and return
        return torch.Tensor(np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))).transpose(0, 1)

    def normalize_audio(self, audio):
        """
        one function to apply them all in an
        order that makes sense.
        """
        audio = to_mono(audio)
        audio = self.normalize_loudness(audio)
        audio = torch.Tensor(audio).to(self.device)
        audio = self.resample(audio)
        if self.cut_silence:
            audio = self.cut_silence_from_audio(audio)
        return audio.to("cpu")

    def visualize_cleaning(self, unclean_audio):
        """
        displays Mel Spectrogram of unclean audio
        and then displays Mel Spectrogram of the
        cleaned version.
        """
        fig, ax = plt.subplots(nrows=2, ncols=1)
        unclean_audio_mono = to_mono(unclean_audio)
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


if __name__ == '__main__':
    import soundfile

    wav, sr = soundfile.read("../audios/test.wav")
    ap = AudioPreprocessor(input_sr=sr, output_sr=16000, cut_silence=True)
    ap.visualize_cleaning(wav)
