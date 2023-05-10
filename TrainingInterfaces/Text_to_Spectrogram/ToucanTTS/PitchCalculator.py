# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Adapted by Florian Lux 2021

import math

import numpy as np
import parselmouth
import torch
import torch.nn.functional as F
from scipy.interpolate import interp1d

from Preprocessing.articulatory_features import get_feature_to_index_lookup


class Parselmouth(torch.nn.Module):
    """
    F0 estimation with Parselmouth https://parselmouth.readthedocs.io/en/stable/index.html
    """

    def __init__(self, fs=16000, n_fft=1024, hop_length=256, f0min=40, f0max=600, use_token_averaged_f0=True,
                 use_continuous_f0=False, use_log_f0=False, reduction_factor=1):
        super().__init__()
        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.frame_period = 1000 * hop_length / fs
        self.f0min = f0min
        self.f0max = f0max
        self.use_token_averaged_f0 = use_token_averaged_f0
        self.use_continuous_f0 = use_continuous_f0
        self.use_log_f0 = use_log_f0
        if use_token_averaged_f0:
            assert reduction_factor >= 1
        self.reduction_factor = reduction_factor

    def output_size(self):
        return 1

    def get_parameters(self):
        return dict(fs=self.fs, n_fft=self.n_fft, hop_length=self.hop_length, f0min=self.f0min, f0max=self.f0max,
                    use_token_averaged_f0=self.use_token_averaged_f0, use_continuous_f0=self.use_continuous_f0, use_log_f0=self.use_log_f0,
                    reduction_factor=self.reduction_factor)

    def forward(self, input_waves, input_waves_lengths=None, feats_lengths=None, durations=None,
                durations_lengths=None, norm_by_average=True, text=None):

        # F0 extraction
        pitch = self._calculate_f0(input_waves[0])

        # Adjust length to match with the mel-spectrogram
        pitch = self._adjust_num_frames(pitch, feats_lengths[0]).view(-1)

        pitch = self._average_by_duration(pitch, durations[0], text).view(-1)
        pitch_lengths = durations_lengths

        if norm_by_average:
            average = pitch[pitch != 0.0].mean()
            pitch = pitch / average

        # Return with the shape (B, T, 1)
        return pitch.unsqueeze(-1), pitch_lengths

    def _calculate_f0(self, input):
        x = input.cpu().numpy().astype(np.double)
        snd = parselmouth.Sound(values=x, sampling_frequency=self.fs)
        f0 = snd.to_pitch(time_step=self.hop_length / self.fs, pitch_floor=self.f0min, pitch_ceiling=self.f0max).selected_array['frequency']
        if self.use_continuous_f0:
            f0 = self._convert_to_continuous_f0(f0)
        if self.use_log_f0:
            nonzero_idxs = np.where(f0 != 0)[0]
            f0[nonzero_idxs] = np.log(f0[nonzero_idxs])
        return input.new_tensor(f0.reshape(-1), dtype=torch.float)

    @staticmethod
    def _adjust_num_frames(x, num_frames):
        if num_frames > len(x):
            # x = F.pad(x, (0, num_frames - len(x)))
            x = F.pad(x, (math.ceil((num_frames - len(x)) / 2), math.floor((num_frames - len(x)) / 2)))
        elif num_frames < len(x):
            x = x[:num_frames]
        return x

    @staticmethod
    def _convert_to_continuous_f0(f0: np.array):
        if (f0 == 0).all():
            return f0

        # padding start and end of f0 sequence
        start_f0 = f0[f0 != 0][0]
        end_f0 = f0[f0 != 0][-1]
        start_idx = np.where(f0 == start_f0)[0][0]
        end_idx = np.where(f0 == end_f0)[0][-1]
        f0[:start_idx] = start_f0
        f0[end_idx:] = end_f0

        # get non-zero frame index
        nonzero_idxs = np.where(f0 != 0)[0]

        # perform linear interpolation
        interp_fn = interp1d(nonzero_idxs, f0[nonzero_idxs])
        f0 = interp_fn(np.arange(0, f0.shape[0]))

        return f0

    def _average_by_duration(self, x, d, text=None):
        d_cumsum = F.pad(d.cumsum(dim=0), (1, 0))
        x_avg = [
            x[start:end].masked_select(x[start:end].gt(0.0)).mean(dim=0) if len(x[start:end].masked_select(x[start:end].gt(0.0))) != 0 else x.new_tensor(0.0)
            for start, end in zip(d_cumsum[:-1], d_cumsum[1:])]

        # find tokens that are not voiced and set pitch to 0
        if text is not None:
            for i, vector in enumerate(text):
                if vector[get_feature_to_index_lookup()["voiced"]] == 0:
                    x_avg[i] = torch.tensor(0.0)

        return torch.stack(x_avg)
