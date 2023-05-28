import numpy
import pyloudnorm as pyln
import torch
from torchaudio.transforms import MelSpectrogram
from torchaudio.transforms import Resample


class AudioPreprocessor:

    def __init__(self, input_sr, output_sr=None, melspec_buckets=128, cut_silence=False, do_loudnorm=False, device="cpu"):
        """
        The parameters are by default set up to do well
        on a 16kHz signal. A different sampling rate may
        require different hop_length and n_fft (e.g.
        doubling frequency --> doubling hop_length and
        doubling n_fft)
        """
        self.cut_silence = cut_silence
        self.do_loudnorm = do_loudnorm
        self.device = device
        self.input_sr = input_sr
        self.output_sr = output_sr
        self.melspec_buckets = melspec_buckets
        self.meter = pyln.Meter(input_sr)
        self.final_sr = input_sr
        self.wave_to_spectrogram = MelSpectrogram(sample_rate=output_sr,
                                                  n_fft=400,
                                                  win_length=400,
                                                  hop_length=200,
                                                  f_min=0.0,
                                                  f_max=output_sr // 2,
                                                  pad=0,
                                                  n_mels=self.melspec_buckets,
                                                  power=2.0,
                                                  normalized=False,
                                                  center=True,
                                                  pad_mode='reflect',
                                                  mel_scale='htk')
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
        if output_sr is not None and output_sr != input_sr:
            self.resample = Resample(orig_freq=input_sr, new_freq=output_sr).to(self.device)
            self.final_sr = output_sr
        else:
            self.resample = lambda x: x

    def cut_leading_and_trailing_silence(self, audio):
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

    def normalize_audio(self, audio):
        """
        one function to apply them all in an
        order that makes sense.
        """
        if self.do_loudnorm:
            audio = self.normalize_loudness(audio)
        audio = torch.Tensor(audio).to(self.device)
        audio = self.resample(audio)
        if self.cut_silence:
            audio = self.cut_leading_and_trailing_silence(audio)
        return audio.to("cpu")

    def audio_to_mel_spec_tensor(self, audio, normalize=True, explicit_sampling_rate=None):
        """
        explicit_sampling_rate is for when
        normalization has already been applied
        and that included resampling. No way
        to detect the current input_sr of the incoming
        audio
        """
        if explicit_sampling_rate is None or explicit_sampling_rate == self.output_sr:
            if normalize:
                audio = self.normalize_audio(audio)
            return self.wave_to_spectrogram(audio)
        print("WARNING: different sampling rate used, this will be very slow if it happens often. Consider creating a dedicated audio processor.")
        return MelSpectrogram(sample_rate=explicit_sampling_rate,
                              n_fft=400,
                              win_length=400,
                              hop_length=200,
                              f_min=0.0,
                              f_max=explicit_sampling_rate // 2,
                              pad=0,
                              n_mels=self.melspec_buckets,
                              power=2.0,
                              normalized=False,
                              center=True,
                              pad_mode='reflect',
                              mel_scale='htk')(audio)


if __name__ == '__main__':
    import soundfile

    wav, sr = soundfile.read("../audios/test.wav")
    ap = AudioPreprocessor(input_sr=sr, output_sr=16000, cut_silence=True)
