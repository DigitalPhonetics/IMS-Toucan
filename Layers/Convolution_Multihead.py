# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Adapted by Florian Lux 2021

import torch
from torch import nn
from torch import Tensor, nn
from typing import Optional, Tuple, Union

class ConvolutionModule_Multihead(nn.Module):
    """
    ConvolutionModule in Conformer model.

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernel size of conv layers.

    """

    def __init__(self, channels, kernel_size, bias=True):
        super(ConvolutionModule_Multihead, self).__init__()
        # kernel_size should be an odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, kernel_size=1, stride=1, padding=0, bias=bias, )
        self.depthwise_conv = nn.Conv1d(channels, channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2, groups=channels, bias=bias, )
        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0, bias=bias, )
        self.activation = Swish()

    def forward(self, x, src_key_padding_mask: Optional[Tensor] = None):
        """
        Compute convolution module.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).

        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).

        """
        # exchange the temporal dimension and the feature dimension
        #x = x.transpose(1, 2)
        x = x.permute(1, 2, 0)  # (#batch, channels, time).


        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        if src_key_padding_mask is not None:
            x.masked_fill_(src_key_padding_mask.unsqueeze(1).expand_as(x), 0.0)
        x = self.depthwise_conv(x)
        if self.use_batchnorm:
            x = self.norm(x)
        x = self.activation(x)

        x = self.pointwise_conv2(x)  # (batch, channel, time)
        #return x.transpose(1, 2)
        return x.permute(2, 0, 1)
        

class Swish(torch.nn.Module):
    """Construct an Swish object."""

    def forward(self, x: Tensor) -> Tensor:
        """Return Swich activation function."""
        return x * torch.sigmoid(x)