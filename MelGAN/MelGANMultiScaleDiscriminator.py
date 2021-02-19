# -*- coding: utf-8 -*-

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""
MelGAN Modules.
"""

import torch
from MelGAN.MelGANDiscriminator import MelGANDiscriminator

class MelGANMultiScaleDiscriminator(torch.nn.Module):
    """
    MelGAN multi-scale discriminator module.
    """

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 scales=3,
                 downsample_pooling="AvgPool1d",
                 # follow the official implementation setting
                 downsample_pooling_params={"kernel_size": 4,
                                            "stride": 2,
                                            "padding": 1,
                                            "count_include_pad": False},
                 kernel_sizes=[5, 3],
                 channels=16,
                 max_downsample_channels=1024,
                 bias=True,
                 downsample_scales=[4, 4, 4, 4],
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.2},
                 pad="ReflectionPad1d",
                 pad_params={},
                 use_weight_norm=True):
        """
        Initilize MelGAN multi-scale discriminator module.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            downsample_pooling (str): Pooling module name for downsampling of the inputs.
            downsample_pooling_params (dict): Parameters for the above pooling module.
            kernel_sizes (list): List of two kernel sizes. The sum will be used for the first conv layer,
                and the first and the second kernel sizes will be used for the last two layers.
            channels (int): Initial number of channels for conv layer.
            max_downsample_channels (int): Maximum number of channels for downsampling layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (list): List of downsampling scales.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (dict): Hyperparameters for padding function.
        """
        super(MelGANMultiScaleDiscriminator, self).__init__()
        self.discriminators = torch.nn.ModuleList()
        for _ in range(scales):
            self.discriminators += [MelGANDiscriminator(in_channels=in_channels,
                                                        out_channels=out_channels,
                                                        kernel_sizes=kernel_sizes,
                                                        channels=channels,
                                                        max_downsample_channels=max_downsample_channels,
                                                        bias=bias,
                                                        downsample_scales=downsample_scales,
                                                        nonlinear_activation=nonlinear_activation,
                                                        nonlinear_activation_params=nonlinear_activation_params,
                                                        pad=pad,
                                                        pad_params=pad_params)]
        self.pooling = getattr(torch.nn, downsample_pooling)(**downsample_pooling_params)
        if use_weight_norm:
            self.apply_weight_norm()
        self.reset_parameters()

    def forward(self, x):
        """
        Calculate forward propagation.
        Args:
            x (Tensor): Input noise signal (B, 1, T).
        Returns:
            List: List of list of each discriminator outputs, which consists of each layer output tensors.
        """
        outs = []
        for f in self.discriminators:
            outs += [f(x)]
            x = self.pooling(x)
        return outs

    def remove_weight_norm(self):
        """
        Remove weight normalization module from all of the layers.
        """

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """
        Apply weight normalization module from all of the layers.
        """

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        """
        Reset parameters.

        This initialization follows official implementation manner.
        https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py
        """

        def _reset_parameters(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                m.weight.data.normal_(0.0, 0.02)

        self.apply(_reset_parameters)
