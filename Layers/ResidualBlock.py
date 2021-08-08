# -*- coding: utf-8 -*-

"""
References:
    - https://github.com/jik876/hifi-gan
    - https://github.com/kan-bayashi/ParallelWaveGAN
"""

import torch


class Conv1d(torch.nn.Conv1d):
    """
    Conv1d module with customized initialization.
    """

    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)


class Conv1d1x1(Conv1d):
    """
    1x1 Conv1d with customized initialization.
    """

    def __init__(self, in_channels, out_channels, bias):
        super(Conv1d1x1, self).__init__(in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=bias)


class HiFiGANResidualBlock(torch.nn.Module):
    """Residual block module in HiFiGAN."""

    def __init__(self,
                 kernel_size=3,
                 channels=512,
                 dilations=(1, 3, 5),
                 bias=True,
                 use_additional_convs=True,
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.1}, ):
        """
        Initialize HiFiGANResidualBlock module.

        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            channels (int): Number of channels for convolution layer.
            dilations (List[int]): List of dilation factors.
            use_additional_convs (bool): Whether to use additional convolution layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
        """
        super().__init__()
        self.use_additional_convs = use_additional_convs
        self.convs1 = torch.nn.ModuleList()
        if use_additional_convs:
            self.convs2 = torch.nn.ModuleList()
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        for dilation in dilations:
            self.convs1 += [torch.nn.Sequential(getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                                                torch.nn.Conv1d(channels,
                                                                channels,
                                                                kernel_size,
                                                                1,
                                                                dilation=dilation,
                                                                bias=bias,
                                                                padding=(kernel_size - 1) // 2 * dilation, ), )]
            if use_additional_convs:
                self.convs2 += [torch.nn.Sequential(getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                                                    torch.nn.Conv1d(channels,
                                                                    channels,
                                                                    kernel_size,
                                                                    1,
                                                                    dilation=1,
                                                                    bias=bias,
                                                                    padding=(kernel_size - 1) // 2, ), )]

    def forward(self, x):
        """
        Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, channels, T).

        Returns:
            Tensor: Output tensor (B, channels, T).
        """
        for idx in range(len(self.convs1)):
            xt = self.convs1[idx](x)
            if self.use_additional_convs:
                xt = self.convs2[idx](xt)
            x = xt + x
        return x
