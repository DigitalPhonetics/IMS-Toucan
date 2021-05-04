import numpy as np
import torch

from Layers.ResidualStack import ResidualStack


class MelGANGenerator(torch.nn.Module):

    def __init__(self,
                 path_to_weights,
                 in_channels=80, out_channels=1, kernel_size=7, channels=512, bias=True,
                 upsample_scales=[8, 8, 2, 2], stack_kernel_size=3, stacks=3,
                 nonlinear_activation="LeakyReLU", nonlinear_activation_params={"negative_slope": 0.2},
                 pad="ReflectionPad1d", pad_params={},
                 use_final_nonlinear_activation=True, use_weight_norm=True):
        super(MelGANGenerator, self).__init__()
        assert channels >= np.prod(upsample_scales)
        assert channels % (2 ** len(upsample_scales)) == 0
        assert (kernel_size - 1) % 2 == 0, "even number for kernel size does not work."
        layers = []
        layers += [getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params),
                   torch.nn.Conv1d(in_channels, channels, kernel_size, bias=bias)]
        for i, upsample_scale in enumerate(upsample_scales):
            layers += [getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)]
            layers += [torch.nn.ConvTranspose1d(channels // (2 ** i), channels // (2 ** (i + 1)), upsample_scale * 2, stride=upsample_scale,
                                                padding=upsample_scale // 2 + upsample_scale % 2, output_padding=upsample_scale % 2, bias=bias)]
            for j in range(stacks):
                layers += [ResidualStack(kernel_size=stack_kernel_size, channels=channels // (2 ** (i + 1)), dilation=stack_kernel_size ** j, bias=bias,
                                         nonlinear_activation=nonlinear_activation, nonlinear_activation_params=nonlinear_activation_params, pad=pad,
                                         pad_params=pad_params)]
        layers += [getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)]
        layers += [getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params),
                   torch.nn.Conv1d(channels // (2 ** (i + 1)), out_channels, kernel_size, bias=bias)]

        if use_final_nonlinear_activation:
            layers += [torch.nn.Tanh()]
        self.melgan = torch.nn.Sequential(*layers)
        if use_weight_norm:
            self.apply_weight_norm()
        self.load_state_dict(torch.load(path_to_weights, map_location='cpu')["generator"])

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

    def forward(self, melspec):
        self.melgan.eval()
        return self.melgan(melspec)
