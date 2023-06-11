# Copyright 2019 Tomoki Hayashi
# MIT License (https://opensource.org/licenses/MIT)
# Adapted by Florian Lux 2021


import torch

from Layers.ConditionalLayerNorm import ConditionalLayerNorm
from Layers.LayerNorm import LayerNorm


class DurationPredictor(torch.nn.Module):
    """
    Duration predictor module.

    This is a module of duration predictor described
    in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The duration predictor predicts a duration of each frame in log domain
    from the hidden embeddings of encoder.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    Note:
        The calculation domain of outputs is different
        between in `forward` and in `inference`. In `forward`,
        the outputs are calculated in log domain but in `inference`,
        those are calculated in linear domain.

    """

    def __init__(self, idim, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1, offset=1.0, utt_embed_dim=None):
        """
        Initialize duration predictor module.

        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
            offset (float, optional): Offset value to avoid nan in log domain.

        """
        super(DurationPredictor, self).__init__()
        self.offset = offset
        self.conv = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2, ),
                                              torch.nn.ReLU())]
            if utt_embed_dim is not None:
                self.norms += [ConditionalLayerNorm(normal_shape=n_chans, speaker_embedding_dim=utt_embed_dim, dim=1)]
            else:
                self.norms += [LayerNorm(n_chans, dim=1)]
            self.dropouts += [torch.nn.Dropout(dropout_rate)]

        self.linear = torch.nn.Linear(n_chans, 1)

    def _forward(self, xs, x_masks=None, is_inference=False, utt_embed=None):
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)

        for f, c, d in zip(self.conv, self.norms, self.dropouts):
            xs = f(xs)  # (B, C, Tmax)
            if utt_embed is not None:
                xs = c(xs, utt_embed)
            else:
                xs = c(xs)
            xs = d(xs)

        # NOTE: targets are transformed to log domain in the loss calculation, so this will learn to predict in the log space, which makes the value range easier to handle.
        xs = self.linear(xs.transpose(1, -1)).squeeze(-1)  # (B, Tmax)

        if is_inference:
            # NOTE: since we learned to predict in the log domain, we have to invert the log during inference.
            xs = torch.clamp(torch.round(xs.exp() - self.offset), min=0).long()  # avoid negative value
        else:
            xs = xs.masked_fill(x_masks, 0.0)

        return xs

    def forward(self, xs, padding_mask=None, utt_embed=None):
        """
        Calculate forward propagation.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            padding_mask (ByteTensor, optional):
                Batch of masks indicating padded part (B, Tmax).

        Returns:
            Tensor: Batch of predicted durations in log domain (B, Tmax).

        """
        return self._forward(xs, padding_mask, False, utt_embed=utt_embed)

    def inference(self, xs, padding_mask=None, utt_embed=None):
        """
        Inference duration.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            padding_mask (ByteTensor, optional):
                Batch of masks indicating padded part (B, Tmax).

        Returns:
            LongTensor: Batch of predicted durations in linear domain (B, Tmax).

        """
        return self._forward(xs, padding_mask, True, utt_embed=utt_embed)


class DurationPredictorLoss(torch.nn.Module):
    """
    Loss function module for duration predictor.

    The loss value is Calculated in log domain to make it Gaussian.

    """

    def __init__(self, offset=1.0, reduction="mean"):
        """
        Args:
            offset (float, optional): Offset value to avoid nan in log domain.
            reduction (str): Reduction type in loss calculation.

        """
        super(DurationPredictorLoss, self).__init__()
        self.criterion = torch.nn.MSELoss(reduction=reduction)
        self.offset = offset

    def forward(self, outputs, targets):
        """
        Calculate forward propagation.

        Args:
            outputs (Tensor): Batch of prediction durations in log domain (B, T)
            targets (LongTensor): Batch of groundtruth durations in linear domain (B, T)

        Returns:
            Tensor: Mean squared error loss value.

        Note:
            `outputs` is in log domain but `targets` is in linear domain.

        """
        # NOTE: outputs is in log domain while targets in linear
        targets = torch.log(targets.float() + self.offset)
        loss = self.criterion(outputs, targets)

        return loss
