"""
MIT licensed code taken and adapted from https://github.com/rishikksh20/Avocodo-pytorch

Copyright (c) 2022 Rishikesh (ऋषिकेश)
adapted 2022, Florian Lux
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal as sig
from torch.nn import Conv1d
from torch.nn.utils import spectral_norm
from torch.nn.utils import weight_norm


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class MultiCoMBDiscriminator(torch.nn.Module):

    def __init__(self, kernels, channels, groups, strides):
        super(MultiCoMBDiscriminator, self).__init__()
        self.combd_1 = CoMBD(filters=channels, kernels=kernels[0], groups=groups, strides=strides)
        self.combd_2 = CoMBD(filters=channels, kernels=kernels[1], groups=groups, strides=strides)
        self.combd_3 = CoMBD(filters=channels, kernels=kernels[2], groups=groups, strides=strides)

        self.pqmf_2 = PQMF(N=2, taps=256, cutoff=0.25, beta=10.0)
        self.pqmf_4 = PQMF(N=8, taps=192, cutoff=0.13, beta=10.0)

    def forward(self, wave_final, intermediate_wave_upsampled_twice=None, intermediate_wave_upsampled_once=None):

        if intermediate_wave_upsampled_twice is not None and intermediate_wave_upsampled_once is not None:
            # get features of generated wave
            features_of_predicted = []

            _, p3_fmap_hat = self.combd_3(wave_final)
            features_of_predicted.append(p3_fmap_hat)

            x2_hat_ = self.pqmf_2(wave_final)[:, :1, :]
            x1_hat_ = self.pqmf_4(wave_final)[:, :1, :]

            _, p2_fmap_hat_ = self.combd_2(intermediate_wave_upsampled_twice)
            features_of_predicted.append(p2_fmap_hat_)

            _, p1_fmap_hat_ = self.combd_1(intermediate_wave_upsampled_once)
            features_of_predicted.append(p1_fmap_hat_)

            _, p2_fmap_hat = self.combd_2(x2_hat_)
            features_of_predicted.append(p2_fmap_hat)

            _, p1_fmap_hat = self.combd_1(x1_hat_)
            features_of_predicted.append(p1_fmap_hat)

            return features_of_predicted

        else:
            # get features of gold wave
            features_of_gold = []

            _, p3_fmap = self.combd_3(wave_final)
            features_of_gold.append(p3_fmap)

            x2_ = self.pqmf_2(wave_final)[:, :1, :]  # Select first band
            x1_ = self.pqmf_4(wave_final)[:, :1, :]  # Select first band

            _, p2_fmap_ = self.combd_2(x2_)
            features_of_gold.append(p2_fmap_)

            _, p1_fmap_ = self.combd_1(x1_)
            features_of_gold.append(p1_fmap_)

            _, p2_fmap = self.combd_2(x2_)
            features_of_gold.append(p2_fmap)

            _, p1_fmap = self.combd_1(x1_)
            features_of_gold.append(p1_fmap)

            return features_of_gold


class MultiSubBandDiscriminator(torch.nn.Module):

    def __init__(self,
                 tkernels,
                 fkernel,
                 tchannels,
                 fchannels,
                 tstrides,
                 fstride,
                 tdilations,
                 fdilations,
                 tsubband,
                 n,
                 m,
                 freq_init_ch):
        super(MultiSubBandDiscriminator, self).__init__()

        self.fsbd = SubBandDiscriminator(init_channel=freq_init_ch, channels=fchannels, kernel=fkernel,
                                         strides=fstride, dilations=fdilations)

        self.tsubband1 = tsubband[0]
        self.tsbd1 = SubBandDiscriminator(init_channel=self.tsubband1, channels=tchannels, kernel=tkernels[0],
                                          strides=tstrides[0], dilations=tdilations[0])

        self.tsubband2 = tsubband[1]
        self.tsbd2 = SubBandDiscriminator(init_channel=self.tsubband2, channels=tchannels, kernel=tkernels[1],
                                          strides=tstrides[1], dilations=tdilations[1])

        self.tsubband3 = tsubband[2]
        self.tsbd3 = SubBandDiscriminator(init_channel=self.tsubband3, channels=tchannels, kernel=tkernels[2],
                                          strides=tstrides[2], dilations=tdilations[2])

        self.pqmf_n = PQMF(N=n, taps=256, cutoff=0.03, beta=10.0)
        self.pqmf_m = PQMF(N=m, taps=256, cutoff=0.1, beta=9.0)

    def forward(self, wave):
        fmap_hat = []

        # Time analysis
        xn_hat = self.pqmf_n(wave)

        q3_hat, feat_q3_hat = self.tsbd3(xn_hat[:, :self.tsubband3, :])
        fmap_hat.append(feat_q3_hat)

        q2_hat, feat_q2_hat = self.tsbd2(xn_hat[:, :self.tsubband2, :])
        fmap_hat.append(feat_q2_hat)

        q1_hat, feat_q1_hat = self.tsbd1(xn_hat[:, :self.tsubband1, :])
        fmap_hat.append(feat_q1_hat)

        # Frequency analysis
        xm_hat = self.pqmf_m(wave)

        xm_hat = xm_hat.transpose(-2, -1)

        q4_hat, feat_q4_hat = self.fsbd(xm_hat)
        fmap_hat.append(feat_q4_hat)

        return fmap_hat


class CoMBD(torch.nn.Module):

    def __init__(self, filters, kernels, groups, strides, use_spectral_norm=False):
        super(CoMBD, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList()
        init_channel = 1
        for i, (f, k, g, s) in enumerate(zip(filters, kernels, groups, strides)):
            self.convs.append(norm_f(Conv1d(init_channel, f, k, s, padding=get_padding(k, 1), groups=g)))
            init_channel = f
        self.conv_post = norm_f(Conv1d(filters[-1], 1, 3, 1, padding=get_padding(3, 1)))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        # fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MDC(torch.nn.Module):

    def __init__(self, in_channel, channel, kernel, stride, dilations, use_spectral_norm=False):
        super(MDC, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = torch.nn.ModuleList()
        self.num_dilations = len(dilations)
        for d in dilations:
            self.convs.append(norm_f(Conv1d(in_channel, channel, kernel, stride=1, padding=get_padding(kernel, d),
                                            dilation=d)))

        self.conv_out = norm_f(Conv1d(channel, channel, 3, stride=stride, padding=get_padding(3, 1)))

    def forward(self, x):
        xs = None
        for l in self.convs:
            if xs is None:
                xs = l(x)
            else:
                xs += l(x)

        x = xs / self.num_dilations

        x = self.conv_out(x)
        x = F.leaky_relu(x, 0.1)
        return x


class SubBandDiscriminator(torch.nn.Module):

    def __init__(self, init_channel, channels, kernel, strides, dilations, use_spectral_norm=False):
        super(SubBandDiscriminator, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm

        self.mdcs = torch.nn.ModuleList()

        for channel, stride, dilation in zip(channels, strides, dilations):
            self.mdcs.append(MDC(init_channel, channel, kernel, stride, dilation))
            init_channel = channel  # output channel of this layer becomes input channel of next layer
        self.conv_post = norm_f(Conv1d(init_channel, 1, 3, padding=get_padding(3, 1)))

    def forward(self, x):
        fmap = []

        for l in self.mdcs:
            x = l(x)
            fmap.append(x)
        x = self.conv_post(x)
        # fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


# adapted from
# https://github.com/kan-bayashi/ParallelWaveGAN/tree/master/parallel_wavegan
class PQMF(torch.nn.Module):

    def __init__(self, N=4, taps=62, cutoff=0.15, beta=9.0):
        super(PQMF, self).__init__()

        self.N = N
        self.taps = taps
        self.cutoff = cutoff
        self.beta = beta

        QMF = sig.firwin(taps + 1, cutoff, window=('kaiser', beta))
        H = np.zeros((N, len(QMF)))
        G = np.zeros((N, len(QMF)))
        for k in range(N):
            constant_factor = (2 * k + 1) * (np.pi /
                                             (2 * N)) * (np.arange(taps + 1) -
                                                         ((taps - 1) / 2))  # TODO: (taps - 1) -> taps
            phase = (-1) ** k * np.pi / 4
            H[k] = 2 * QMF * np.cos(constant_factor + phase)

            G[k] = 2 * QMF * np.cos(constant_factor - phase)

        H = torch.from_numpy(H[:, None, :]).float()
        G = torch.from_numpy(G[None, :, :]).float()

        self.register_buffer("H", H)
        self.register_buffer("G", G)

        updown_filter = torch.zeros((N, N, N)).float()
        for k in range(N):
            updown_filter[k, k, 0] = 1.0
        self.register_buffer("updown_filter", updown_filter)
        self.N = N

        self.pad_fn = torch.nn.ConstantPad1d(taps // 2, 0.0)

    def forward(self, x):
        return self.analysis(x)

    def analysis(self, x):
        return F.conv1d(x, self.H, padding=self.taps // 2, stride=self.N)

    def synthesis(self, x):
        x = F.conv_transpose1d(x,
                               self.updown_filter * self.N,
                               stride=self.N)
        x = F.conv1d(x, self.G, padding=self.taps // 2)
        return x
