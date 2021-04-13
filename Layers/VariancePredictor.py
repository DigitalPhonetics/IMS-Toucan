# Copyright 2019 Tomoki Hayashi
# MIT License (https://opensource.org/licenses/MIT)
# Adapted by Florian Lux 2021

from abc import ABC

import torch

from Layers.LayerNorm import LayerNorm


class VariancePredictor(torch.nn.Module, ABC):
    """
    Variance predictor module.

    This is a module of variance predictor described in `FastSpeech 2:
    Fast and High-Quality End-to-End Text to Speech`_.

    .. _`FastSpeech 2: Fast and High-Quality End-to-End Text to Speech`:
        https://arxiv.org/abs/2006.04558

    """

    def __init__(self, idim, n_layers=2, n_chans=384, kernel_size=3, bias=True, dropout_rate=0.5, ):
        """
        Initilize duration predictor module.

        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
        """
        super().__init__()
        self.conv = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [
                torch.nn.Sequential(torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias, ), torch.nn.ReLU(),
                                    LayerNorm(n_chans, dim=1), torch.nn.Dropout(dropout_rate), )]
        self.linear = torch.nn.Linear(n_chans, 1)

    def forward(self, xs, x_masks=None):
        """
        Calculate forward propagation.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional):
                Batch of masks indicating padded part (B, Tmax).

        Returns:
            Tensor: Batch of predicted sequences (B, Tmax, 1).
        """
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)

        xs = self.linear(xs.transpose(1, 2))  # (B, Tmax, 1)

        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0.0)

        return xs
