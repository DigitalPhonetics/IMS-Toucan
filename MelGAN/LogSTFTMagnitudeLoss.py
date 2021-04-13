# Copyright 2019 Tomoki Hayashi
# MIT License (https://opensource.org/licenses/MIT)
# Adapted by Florian Lux 2021


import torch
import torch.nn.functional as F


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """
    Log STFT magnitude loss module.
    """

    def __init__(self):
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """
        Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))
