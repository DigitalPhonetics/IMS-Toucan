# -*- coding: utf-8 -*-

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""
MelGAN Modules.
"""

import numpy as np
import torch


class MelGANDiscriminator(torch.nn.Module):
    """
    MelGAN discriminator module.
    """

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 kernel_sizes=[5, 3],
                 channels=16,
                 max_downsample_channels=512,
                 bias=True,
                 downsample_scales=[4, 4, 4],
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.2},
                 pad="ReflectionPad1d",
                 pad_params={}):
        """
        Initilize MelGAN discriminator module.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_sizes (list): List of two kernel sizes. The prod will be used for the first conv layer,
                and the first and the second kernel sizes will be used for the last two layers.
                For example if kernel_sizes = [5, 3], the first layer kernel size will be 5 * 3 = 15,
                the last two layers' kernel size will be 5 and 3, respectively.
            channels (int): Initial number of channels for conv layer.
            max_downsample_channels (int): Maximum number of channels for downsampling layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (list): List of downsampling scales.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (dict): Hyperparameters for padding function.
        """
        super(MelGANDiscriminator, self).__init__()
        self.layers = torch.nn.ModuleList()
        # check kernel size is valid
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1
        assert kernel_sizes[1] % 2 == 1
        # add first layer
        self.layers += [torch.nn.Sequential(getattr(torch.nn, pad)((np.prod(kernel_sizes) - 1) // 2, **pad_params),
                                            torch.nn.Conv1d(in_channels, channels, np.prod(kernel_sizes), bias=bias),
                                            getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params))]
        # add downsample layers
        in_chs = channels
        for downsample_scale in downsample_scales:
            out_chs = min(in_chs * downsample_scale, max_downsample_channels)
            self.layers += [torch.nn.Sequential(torch.nn.Conv1d(in_chs,
                                                                out_chs,
                                                                kernel_size=downsample_scale * 10 + 1,
                                                                stride=downsample_scale,
                                                                padding=downsample_scale * 5,
                                                                groups=in_chs // 4,
                                                                bias=bias),
                                                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params))]
            in_chs = out_chs

        # add final layers
        out_chs = min(in_chs * 2, max_downsample_channels)
        self.layers += [torch.nn.Sequential(torch.nn.Conv1d(in_chs,
                                                            out_chs,
                                                            kernel_sizes[0],
                                                            padding=(kernel_sizes[0] - 1) // 2,
                                                            bias=bias),
                                            getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params))]
        self.layers += [torch.nn.Conv1d(out_chs, out_channels, kernel_sizes[1],
                                        padding=(kernel_sizes[1] - 1) // 2,
                                        bias=bias)]

    def forward(self, x):
        """
        Calculate forward propagation.
        Args:
            x (Tensor): Input noise signal (B, 1, T).
        Returns:
            List: List of output tensors of each layer.
        """
        outs = []
        for f in self.layers:
            x = f(x)
            outs += [x]
        return outs
