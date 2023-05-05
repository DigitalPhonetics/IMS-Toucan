# Copyright 2019 Tomoki Hayashi
# MIT License (https://opensource.org/licenses/MIT)
# Adapted by Florian Lux 2023

from abc import ABC

import torch

from Layers.LayerNorm import LayerNorm
from Utility.utils import integrate_with_utt_embed


class VariancePredictor(torch.nn.Module, ABC):
    """
    Variance predictor module.

    This is a module of variance predictor described in `FastSpeech 2:
    Fast and High-Quality End-to-End Text to Speech`_.

    .. _`FastSpeech 2: Fast and High-Quality End-to-End Text to Speech`:
        https://arxiv.org/abs/2006.04558

    """

    def __init__(self,
                 idim,
                 n_layers=2,
                 n_chans=384,
                 kernel_size=3,
                 bias=True,
                 dropout_rate=0.5,
                 utt_embed_dim=None,
                 train_utt_embs=False):
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
        self.embedding_projections = torch.nn.ModuleList()
        self.utt_embed_dim = utt_embed_dim
        self.train_utt_embs = train_utt_embs

        for idx in range(n_layers):
            if utt_embed_dim is not None:
                if train_utt_embs:
                    self.embedding_projections += [torch.nn.Linear(utt_embed_dim, idim)]
                else:
                    self.embedding_projections += [torch.nn.Linear(utt_embed_dim + idim, idim)]
            else:
                self.embedding_projections += [lambda x: x]
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias, ),
                                              torch.nn.ReLU())]
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

        for f, c, d, p in zip(self.conv, self.norms, self.dropouts, self.embedding_projections):
            xs = f(xs)  # (B, C, Tmax)
            if self.utt_embed_dim is not None:
                xs = integrate_with_utt_embed(hs=xs.transpose(1, 2), utt_embeddings=utt_embed, projection=p, embedding_training=self.train_utt_embs).transpose(1, 2)
            xs = c(xs)
            xs = d(xs)

        xs = self.linear(xs.transpose(1, 2))  # (B, Tmax, 1)

        if padding_mask is not None:
            xs = xs.masked_fill(padding_mask, 0.0)

        return xs
