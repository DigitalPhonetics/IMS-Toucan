# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""STFT-based Loss modules."""

import torch


class SpectralConvergenceLoss(torch.nn.Module):
    """
    Spectral convergence loss module.
    """

    def __init__(self):
        """
        Initilize spectral convergence loss module.
        """
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """
        Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")
