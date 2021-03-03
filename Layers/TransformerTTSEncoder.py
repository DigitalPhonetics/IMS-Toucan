# Written by Shigeki Karita, 2019
# Published under Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Adapted by Florian Lux, 2021

"""Encoder definition."""

import torch

from Layers.Attention import MultiHeadedAttention
from Layers.LayerNorm import LayerNorm
from Layers.MultiLayeredConv1d import MultiLayeredConv1d
from Layers.MultiSequential import repeat
from Layers.PositionalEncoding import PositionalEncoding
from Layers.TransformerTTSEncoderLayer import EncoderLayer


class Encoder(torch.nn.Module):
    """Transformer encoder module.

    Args:
        idim (int): Input dimension.
        attention_dim (int): Dimention of attention.
        attention_heads (int): The number of heads of multi head attention.
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        attention_dropout_rate (float): Dropout rate in attention.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        input_layer (Union[str, torch.nn.Module]): Input layer type.
        pos_enc_class (torch.nn.Module): Positional encoding module class.
            `PositionalEncoding `or `ScaledPositionalEncoding`
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
        positionwise_conv_kernel_size (int): Kernel size of positionwise conv1d layer.
        padding_idx (int): Padding idx for input_layer=embed.

    """

    def __init__(
            self,
            idim,
            selfattention_layer_type="selfattn",
            attention_dim=256,
            attention_heads=4,
            conv_wshare=4,
            conv_kernel_length=11,
            conv_usebias=False,
            linear_units=2048,
            num_blocks=6,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.0,
            input_layer="conv2d",
            pos_enc_class=PositionalEncoding,
            normalize_before=True,
            concat_after=False,
            positionwise_layer_type="linear",
            positionwise_conv_kernel_size=1,
            padding_idx=-1,
    ):
        """Construct an Encoder object."""
        super(Encoder, self).__init__()

        self.conv_subsampling_factor = 1
        self.embed = torch.nn.Sequential(
            input_layer,
            pos_enc_class(attention_dim, positional_dropout_rate),
        )
        self.normalize_before = normalize_before
        positionwise_layer, positionwise_layer_args = self.get_positionwise_layer(
            positionwise_layer_type,
            attention_dim,
            linear_units,
            dropout_rate,
            positionwise_conv_kernel_size,
        )
        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads, attention_dim, attention_dropout_rate
                ),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

    def get_positionwise_layer(
            self,
            positionwise_layer_type="linear",
            attention_dim=256,
            linear_units=2048,
            dropout_rate=0.1,
            positionwise_conv_kernel_size=1,
    ):
        """Define positionwise layer."""
        positionwise_layer = MultiLayeredConv1d
        positionwise_layer_args = (
            attention_dim,
            linear_units,
            positionwise_conv_kernel_size,
            dropout_rate,
        )
        return positionwise_layer, positionwise_layer_args

    def forward(self, xs, masks):
        """Encode input sequence.

        Args:
            xs (torch.Tensor): Input tensor (#batch, time, idim).
            masks (torch.Tensor): Mask tensor (#batch, time).

        Returns:
            torch.Tensor: Output tensor (#batch, time, attention_dim).
            torch.Tensor: Mask tensor (#batch, time).

        """
        xs = self.embed(xs)
        xs, masks = self.encoders(xs, masks)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks

    def forward_one_step(self, xs, masks, cache=None):
        """Encode input frame.

        Args:
            xs (torch.Tensor): Input tensor.
            masks (torch.Tensor): Mask tensor.
            cache (List[torch.Tensor]): List of cache tensors.

        Returns:
            torch.Tensor: Output tensor.
            torch.Tensor: Mask tensor.
            List[torch.Tensor]: List of new cache tensors.

        """
        xs = self.embed(xs)
        if cache is None:
            cache = [None for _ in range(len(self.encoders))]
        new_cache = []
        for c, e in zip(cache, self.encoders):
            xs, masks = e(xs, masks, cache=c)
            new_cache.append(xs)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks, new_cache
