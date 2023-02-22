# Copyright (c) 2022 NVIDIA CORPORATION.
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

import torch
from alias_free_torch import Activation1d
from torch.nn import Conv1d
from torch.nn import ConvTranspose1d
from torch.nn import ModuleList
from torch.nn.utils import remove_weight_norm
from torch.nn.utils import weight_norm

from TrainingInterfaces.Spectrogram_to_Wave.BigVGAN.AMP import AMPBlock1
from TrainingInterfaces.Spectrogram_to_Wave.BigVGAN.Snake import SnakeBeta


class BigVGAN(torch.nn.Module):
    # this is the main BigVGAN model. Applies anti-aliased periodic activation for resblocks.

    def __init__(self,
                 num_mels=80,
                 upsample_initial_channel=512,
                 upsample_rates=(8, 6, 4, 2),  # CAREFUL: Avocodo discriminator assumes that there are always 4 upsample scales, because it takes intermediate results.
                 upsample_kernel_sizes=(16, 12, 8, 4),
                 resblock_kernel_sizes=(3, 7, 11),
                 resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
                 ):
        super(BigVGAN, self).__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        # pre conv
        self.conv_pre = weight_norm(Conv1d(num_mels, upsample_initial_channel, 7, 1, padding=3))

        # transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(ModuleList([
                weight_norm(ConvTranspose1d(upsample_initial_channel // (2 ** i),
                                            upsample_initial_channel // (2 ** (i + 1)),
                                            k, u, padding=(k - u) // 2))
            ]))

        # residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(AMPBlock1(ch, k, d))

        # post conv
        activation_post = SnakeBeta(ch, alpha_logscale=True)
        self.activation_post = Activation1d(activation=activation_post)

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))

        # weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

        # for Avocodo discriminator
        self.out_proj_x1 = torch.nn.Conv1d(512 // 4, 1, 7, 1, padding=3)
        self.out_proj_x2 = torch.nn.Conv1d(512 // 8, 1, 7, 1, padding=3)

    def forward(self, x):
        # pre conv
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            # upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
            if i == 1:
                x1 = self.out_proj_x1(x)
            elif i == 2:
                x2 = self.out_proj_x2(x)

        # post conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x, x2, x1

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            for l_i in l:
                remove_weight_norm(l_i)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


if __name__ == '__main__':
    print(BigVGAN()(torch.randn([1, 80, 100]))[0].shape)
