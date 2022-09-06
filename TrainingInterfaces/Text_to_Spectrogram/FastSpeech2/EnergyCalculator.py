# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Adapted by Florian Lux 2021

import torch
import torch.nn.functional as F

from Layers.STFT import STFT
from Preprocessing.articulatory_features import get_feature_to_index_lookup
from Utility.utils import pad_list


class EnergyCalculator(torch.nn.Module):

    def __init__(self, fs=16000, n_fft=1024, win_length=None, hop_length=256, window="hann", center=True,
                 normalized=False, onesided=True, use_token_averaged_energy=True, reduction_factor=1):
        super().__init__()

        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.use_token_averaged_energy = use_token_averaged_energy
        if use_token_averaged_energy:
            assert reduction_factor >= 1
        self.reduction_factor = reduction_factor

        self.stft = STFT(n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window, center=center, normalized=normalized, onesided=onesided)

    def output_size(self):
        return 1

    def get_parameters(self):
        return dict(fs=self.fs, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, win_length=self.win_length, center=self.stft.center,
                    normalized=self.stft.normalized, use_token_averaged_energy=self.use_token_averaged_energy, reduction_factor=self.reduction_factor)

    def forward(self, input_waves, input_waves_lengths=None, feats_lengths=None, durations=None,
                durations_lengths=None, norm_by_average=True, text=None):
        # If not provided, we assume that the inputs have the same length
        if input_waves_lengths is None:
            input_waves_lengths = (input_waves.new_ones(input_waves.shape[0], dtype=torch.long) * input_waves.shape[1])

        # Domain-conversion: e.g. Stft: time -> time-freq
        input_stft, energy_lengths = self.stft(input_waves, input_waves_lengths)

        assert input_stft.dim() >= 4, input_stft.shape
        assert input_stft.shape[-1] == 2, input_stft.shape

        # input_stft: (..., F, 2) -> (..., F)
        input_power = input_stft[..., 0] ** 2 + input_stft[..., 1] ** 2
        # sum over frequency (B, N, F) -> (B, N)
        energy = torch.sqrt(torch.clamp(input_power.sum(dim=2), min=1.0e-10))

        # (Optional): Adjust length to match with the mel-spectrogram
        if feats_lengths is not None:
            energy = [self._adjust_num_frames(e[:el].view(-1), fl) for e, el, fl in zip(energy, energy_lengths, feats_lengths)]
            energy_lengths = feats_lengths

        # (Optional): Average by duration to calculate token-wise energy
        if self.use_token_averaged_energy:
            energy = [self._average_by_duration(e[:el].view(-1), d, text) for e, el, d in zip(energy, energy_lengths, durations)]
            energy_lengths = durations_lengths

        # Padding
        if isinstance(energy, list):
            energy = pad_list(energy, 0.0)

        # Return with the shape (B, T, 1)
        if norm_by_average:
            average = energy[0][energy[0] != 0.0].mean()
            energy = energy / average
        return energy.unsqueeze(-1), energy_lengths

    def _average_by_duration(self, x, d, text=None):
        d_cumsum = F.pad(d.cumsum(dim=0), (1, 0))
        x_avg = [x[start:end].mean() if len(x[start:end]) != 0 else x.new_tensor(0.0) for start, end in zip(d_cumsum[:-1], d_cumsum[1:])]

        # find tokens that are not phoneme and set energy to 0
        if text is not None:
            for i, vector in enumerate(text):
                if vector[get_feature_to_index_lookup()["phoneme"]] == 0:
                    x_avg[i] = torch.tensor(0.0)

        return torch.stack(x_avg)

    @staticmethod
    def _adjust_num_frames(x, num_frames):
        if num_frames > len(x):
            x = F.pad(x, (0, num_frames - len(x)))
        elif num_frames < len(x):
            x = x[:num_frames]
        return x
