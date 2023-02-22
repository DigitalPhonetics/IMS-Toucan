import numpy as np
import torch
import torch.utils.data
import torch.utils.data.distributed
from torch import nn


class ResNet_G(nn.Module):

    def __init__(self, data_dim, z_dim, size, nfilter=64, nfilter_max=512, bn=True, res_ratio=0.1, **kwargs):
        super().__init__()
        self.input_dim = z_dim
        self.output_dim = z_dim
        self.dropout_rate = 0

        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max
        self.bn = bn
        self.z_dim = z_dim

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2 ** (nlayers + 1))

        self.fc = nn.Linear(z_dim, self.nf0 * s0 * s0)
        if self.bn:
            self.bn1d = nn.BatchNorm1d(self.nf0 * s0 * s0)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        blocks = []
        for i in range(nlayers, 0, -1):
            nf0 = min(nf * 2 ** (i + 1), nf_max)
            nf1 = min(nf * 2 ** i, nf_max)
            blocks += [
                ResNetBlock(nf0, nf1, bn=self.bn, res_ratio=res_ratio),
                nn.Upsample(scale_factor=2)
            ]

        nf0 = min(nf * 2, nf_max)
        nf1 = min(nf, nf_max)
        blocks += [
            ResNetBlock(nf0, nf1, bn=self.bn, res_ratio=res_ratio),
            ResNetBlock(nf1, nf1, bn=self.bn, res_ratio=res_ratio)
        ]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

        self.fc_out = nn.Linear(3 * size * size, data_dim)

    def forward(self, z, return_intermediate=False):
        # print(z.shape)
        batch_size = z.size(0)
        # z = z.view(batch_size, -1)
        out = self.fc(z)
        if self.bn:
            out = self.bn1d(out)
        out = self.relu(out)
        if return_intermediate:
            l_1 = out.detach().clone()
        out = out.view(batch_size, self.nf0, self.s0, self.s0)
        # print(out.shape)

        out = self.resnet(out)

        # print(out.shape)
        # out = out.view(batch_size, self.nf0*self.s0*self.s0*2)

        out = self.conv_img(out)
        out = self.relu(out)
        out.flatten(1)
        out = self.fc_out(out.flatten(1))

        if return_intermediate:
            return out, l_1
        return out

    def sample_latent(self, n_samples, z_size):
        return torch.randn((n_samples, z_size))


class ResNet_D(nn.Module):

    def __init__(self, data_dim, size, nfilter=64, nfilter_max=512, res_ratio=0.1):
        super().__init__()
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max
        self.size = size

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2 ** nlayers)

        nf0 = min(nf, nf_max)
        nf1 = min(nf * 2, nf_max)
        blocks = [
            ResNetBlock(nf0, nf0, bn=False, res_ratio=res_ratio),
            ResNetBlock(nf0, nf1, bn=False, res_ratio=res_ratio)
        ]

        self.fc_input = nn.Linear(data_dim, 3 * size * size)

        for i in range(1, nlayers + 1):
            nf0 = min(nf * 2 ** i, nf_max)
            nf1 = min(nf * 2 ** (i + 1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResNetBlock(nf0, nf1, bn=False, res_ratio=res_ratio),
            ]

        self.conv_img = nn.Conv2d(3, 1 * nf, 3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.resnet = nn.Sequential(*blocks)

        self.fc = nn.Linear(self.nf0 * s0 * s0, 1)

    def forward(self, x):
        batch_size = x.size(0)

        out = self.fc_input(x)
        out = self.relu(out).view(batch_size, 3, self.size, self.size)

        out = self.relu((self.conv_img(out)))
        out = self.resnet(out)
        out = out.view(batch_size, self.nf0 * self.s0 * self.s0)
        out = self.fc(out)

        return out


class ResNetBlock(nn.Module):

    def __init__(self, fin, fout, fhidden=None, bn=True, res_ratio=0.1):
        super().__init__()
        # Attributes
        self.bn = bn
        self.is_bias = not bn
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden
        self.res_ratio = res_ratio

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1, bias=self.is_bias)
        if self.bn:
            self.bn2d_0 = nn.BatchNorm2d(self.fhidden)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=self.is_bias)
        if self.bn:
            self.bn2d_1 = nn.BatchNorm2d(self.fout)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)
            if self.bn:
                self.bn2d_s = nn.BatchNorm2d(self.fout)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(x)
        if self.bn:
            dx = self.bn2d_0(dx)
        dx = self.relu(dx)
        dx = self.conv_1(dx)
        if self.bn:
            dx = self.bn2d_1(dx)
        out = self.relu(x_s + self.res_ratio * dx)
        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
            if self.bn:
                x_s = self.bn2d_s(x_s)
        else:
            x_s = x
        return x_s
