import torch

from Layers.ResidualBlock import HiFiGANResidualBlock as ResidualBlock


class HiFiGANGenerator(torch.nn.Module):

    def __init__(self,
                 path_to_weights,
                 in_channels=80,
                 out_channels=1,
                 channels=512,
                 kernel_size=7,
                 upsample_scales=(8, 6, 4, 2),
                 upsample_kernel_sizes=(16, 12, 8, 4),
                 resblock_kernel_sizes=(3, 7, 11),
                 resblock_dilations=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
                 use_additional_convs=True,
                 bias=True,
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.1},
                 use_weight_norm=True, ):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        assert len(upsample_scales) == len(upsample_kernel_sizes)
        assert len(resblock_dilations) == len(resblock_kernel_sizes)
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
            self.upsamples += [
                torch.nn.Sequential(getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
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
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(channels // (2 ** (i + 1)),
                            out_channels,
                            kernel_size,
                            1,
                            padding=(kernel_size - 1) // 2, ),
            torch.nn.Tanh(), )

        self.out_proj_x1 = torch.nn.Conv1d(512 // 4, 1, 7, 1, padding=3)
        self.out_proj_x2 = torch.nn.Conv1d(512 // 8, 1, 7, 1, padding=3)

        if use_weight_norm:
            self.apply_weight_norm()

        self.load_state_dict(torch.load(path_to_weights, map_location='cpu')["generator"])

    def forward(self, c, normalize_before=False):
        if normalize_before:
            c = (c - self.mean) / self.scale
        c = self.input_conv(c.unsqueeze(0))
        for i in range(self.num_upsamples):
            c = self.upsamples[i](c)
            cs = 0.0  # initialize
            for j in range(self.num_blocks):
                cs = cs + self.blocks[i * self.num_blocks + j](c)
            c = cs / self.num_blocks
        c = self.output_conv(c)
        return c.squeeze()

    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)
