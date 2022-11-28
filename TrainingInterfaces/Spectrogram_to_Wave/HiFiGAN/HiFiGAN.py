# Copyright 2021 Tomoki Hayashi
# MIT License (https://opensource.org/licenses/MIT)
# Adapted by Florian Lux 2021

# This code is based on https://github.com/jik876/hifi-gan.


import torch

from Layers.ResidualBlock import HiFiGANResidualBlock as ResidualBlock


class HiFiGANGenerator(torch.nn.Module):

    def __init__(self,
                 in_channels=80,
                 out_channels=1,
                 channels=512,
                 kernel_size=7,
                 upsample_scales=(8, 6, 4, 2),  # CAREFUL: Avocodo assumes that there are always 4 upsample scales, because it takes intermediate results.
                 upsample_kernel_sizes=(16, 12, 8, 4),
                 resblock_kernel_sizes=(3, 7, 11),
                 resblock_dilations=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
                 use_additional_convs=True,
                 bias=True,
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.1}):
        """
        Initialize HiFiGANGenerator module.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            channels (int): Number of hidden representation channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            upsample_scales (list): List of upsampling scales.
            upsample_kernel_sizes (list): List of kernel sizes for upsampling layers.
            resblock_kernel_sizes (list): List of kernel sizes for residual blocks.
            resblock_dilations (list): List of dilation list for residual blocks.
            use_additional_convs (bool): Whether to use additional conv layers in residual blocks.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
        """
        super().__init__()

        # check hyperparameters are valid
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        assert len(upsample_scales) == len(upsample_kernel_sizes)
        assert len(resblock_dilations) == len(resblock_kernel_sizes)

        # define modules
        self.num_upsamples = len(upsample_kernel_sizes)
        self.num_blocks = len(resblock_kernel_sizes)
        self.input_conv = torch.nn.Conv1d(in_channels,
                                          channels,
                                          kernel_size,
                                          1,
                                          padding=(kernel_size - 1) // 2, )
        self.upsamples = torch.nn.ModuleList()
        self.blocks = torch.nn.ModuleList()
        for i in range(len(upsample_kernel_sizes)):
            self.upsamples += [torch.nn.Sequential(getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                                                   torch.nn.ConvTranspose1d(channels // (2 ** i),
                                                                            channels // (2 ** (i + 1)),
                                                                            upsample_kernel_sizes[i],
                                                                            upsample_scales[i],
                                                                            padding=(upsample_kernel_sizes[i] - upsample_scales[i]) // 2, ), )]
            for j in range(len(resblock_kernel_sizes)):
                self.blocks += [ResidualBlock(kernel_size=resblock_kernel_sizes[j],
                                              channels=channels // (2 ** (i + 1)),
                                              dilations=resblock_dilations[j],
                                              bias=bias,
                                              use_additional_convs=use_additional_convs,
                                              nonlinear_activation=nonlinear_activation,
                                              nonlinear_activation_params=nonlinear_activation_params, )]
        self.output_conv = torch.nn.Sequential(
            # NOTE(kan-bayashi): follow official implementation but why
            #   using different slope parameter here? (0.1 vs. 0.01)
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(channels // (2 ** (i + 1)),
                            out_channels,
                            kernel_size,
                            1,
                            padding=(kernel_size - 1) // 2, ), torch.nn.Tanh(), )

        self.out_proj_x1 = torch.nn.Conv1d(512 // 4, 1, 7, 1, padding=3)
        self.out_proj_x2 = torch.nn.Conv1d(512 // 8, 1, 7, 1, padding=3)

        # apply weight norm
        self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, c):
        """
        Calculate forward propagation.
        
        Args:
            c (Tensor): Input tensor (B, in_channels, T).
            
        Returns:
            Tensor: Output tensor (B, out_channels, T).
            Tensor: intermediate result
            Tensor: another intermediate result
        """
        c = self.input_conv(c)
        for i in range(self.num_upsamples):
            c = self.upsamples[i](c)
            cs = 0.0  # initialize
            for j in range(self.num_blocks):
                cs += self.blocks[i * self.num_blocks + j](c)
            c = cs / self.num_blocks
            if i == 1:
                x1 = self.out_proj_x1(c)
            elif i == 2:
                x2 = self.out_proj_x2(c)
        c = self.output_conv(c)

        return c, x2, x1

    def reset_parameters(self):
        """
        Reset parameters.
        
        This initialization follows the official implementation manner.
        https://github.com/jik876/hifi-gan/blob/master/models.py
        """

        def _reset_parameters(m):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                m.weight.data.normal_(0.0, 0.01)

        self.apply(_reset_parameters)

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

    def inference(self, c, normalize_before=False):
        """
        Perform inference.
        
        Args:
            c (Union[Tensor, ndarray]): Input tensor (T, in_channels).
            normalize_before (bool): Whether to perform normalization.
            
        Returns:
            Tensor: Output tensor (T ** prod(upsample_scales), out_channels).
        """
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float).to(next(self.parameters()).device)
        if normalize_before:
            c = (c - self.mean) / self.scale
        c = self.forward(c.transpose(1, 0).unsqueeze(0))
        return c.squeeze(0).transpose(1, 0)
