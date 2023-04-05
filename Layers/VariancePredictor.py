# Copyright 2019 Tomoki Hayashi
# MIT License (https://opensource.org/licenses/MIT)
# Adapted by Florian Lux 2023

from abc import ABC

import torch

from Layers.ConditionalLayerNorm import ConditionalLayerNorm
from Layers.LayerNorm import LayerNorm


class VariancePredictor(torch.nn.Module, ABC):
    """
    Variance predictor module.

    This is a module of variance predictor described in `FastSpeech 2:
    Fast and High-Quality End-to-End Text to Speech`_.

    .. _`FastSpeech 2: Fast and High-Quality End-to-End Text to Speech`:
        https://arxiv.org/abs/2006.04558

    """

    def __init__(self, idim, n_layers=2, n_chans=384, kernel_size=3, bias=True, dropout_rate=0.5, utt_embed_dim=None):
        """
        Initialize duration predictor module.

        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
        """
        super().__init__()
        self.conv = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias, ),
                                              torch.nn.ReLU())]
            if utt_embed_dim is not None:
                self.norms += [ConditionalLayerNorm(normal_shape=n_chans, speaker_embedding_dim=utt_embed_dim, dim=1)]
            else:
                self.norms += [LayerNorm(n_chans, dim=1)]
            self.dropouts += [torch.nn.Dropout(dropout_rate)]

        self.linear = torch.nn.Linear(n_chans, 1)

    def forward(self, xs, padding_mask=None, utt_embed=None):
        """
        Calculate forward propagation.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            padding_mask (ByteTensor, optional):
                Batch of masks indicating padded part (B, Tmax).

        Returns:
            Tensor: Batch of predicted sequences (B, Tmax, 1).
        """
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)

        for f, c, d in zip(self.conv, self.norms, self.dropouts):
            xs = f(xs)  # (B, C, Tmax)
            if utt_embed is not None:
                xs = c(xs, utt_embed)
            else:
                xs = c(xs)
            xs = d(xs)

        xs = self.linear(xs.transpose(1, 2))  # (B, Tmax, 1)

        if padding_mask is not None:
            xs = xs.masked_fill(padding_mask, 0.0)

        return xs
