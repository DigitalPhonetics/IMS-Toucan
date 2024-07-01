"""
Code taken from https://github.com/tuanh123789/AdaSpeech/blob/main/model/adaspeech_modules.py
By https://github.com/tuanh123789
No license specified

Implemented as outlined in AdaSpeech https://arxiv.org/pdf/2103.00993.pdf
Used in this toolkit similar to how it is done in AdaSpeech 4 https://arxiv.org/pdf/2204.00436.pdf

"""

import torch
from torch import nn


class ConditionalLayerNorm(nn.Module):

    def __init__(self,
                 hidden_dim,
                 speaker_embedding_dim,
                 dim=-1):
        super(ConditionalLayerNorm, self).__init__()
        self.dim = dim
        if isinstance(hidden_dim, int):
            self.normal_shape = hidden_dim
        self.speaker_embedding_dim = speaker_embedding_dim
        self.W_scale = nn.Sequential(nn.Linear(self.speaker_embedding_dim, self.normal_shape),
                                     nn.Tanh(),
                                     nn.Linear(self.normal_shape, self.normal_shape))
        self.W_bias = nn.Sequential(nn.Linear(self.speaker_embedding_dim, self.normal_shape),
                                    nn.Tanh(),
                                    nn.Linear(self.normal_shape, self.normal_shape))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.W_scale[0].weight, 0.0)
        torch.nn.init.constant_(self.W_scale[2].weight, 0.0)
        torch.nn.init.constant_(self.W_scale[0].bias, 1.0)
        torch.nn.init.constant_(self.W_scale[2].bias, 1.0)
        torch.nn.init.constant_(self.W_bias[0].weight, 0.0)
        torch.nn.init.constant_(self.W_bias[2].weight, 0.0)
        torch.nn.init.constant_(self.W_bias[0].bias, 0.0)
        torch.nn.init.constant_(self.W_bias[2].bias, 0.0)

    def forward(self, x, speaker_embedding):

        if self.dim != -1:
            x = x.transpose(-1, self.dim)

        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        scale = self.W_scale(speaker_embedding)
        bias = self.W_bias(speaker_embedding)

        y = scale.unsqueeze(1) * ((x - mean) / var) + bias.unsqueeze(1)

        if self.dim != -1:
            y = y.transpose(-1, self.dim)

        return y


class SequentialWrappableConditionalLayerNorm(nn.Module):

    def __init__(self,
                 hidden_dim,
                 speaker_embedding_dim):
        super(SequentialWrappableConditionalLayerNorm, self).__init__()
        if isinstance(hidden_dim, int):
            self.normal_shape = hidden_dim
        self.speaker_embedding_dim = speaker_embedding_dim
        self.W_scale = nn.Sequential(nn.Linear(self.speaker_embedding_dim, self.normal_shape),
                                     nn.Tanh(),
                                     nn.Linear(self.normal_shape, self.normal_shape))
        self.W_bias = nn.Sequential(nn.Linear(self.speaker_embedding_dim, self.normal_shape),
                                    nn.Tanh(),
                                    nn.Linear(self.normal_shape, self.normal_shape))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.W_scale[0].weight, 0.0)
        torch.nn.init.constant_(self.W_scale[2].weight, 0.0)
        torch.nn.init.constant_(self.W_scale[0].bias, 1.0)
        torch.nn.init.constant_(self.W_scale[2].bias, 1.0)
        torch.nn.init.constant_(self.W_bias[0].weight, 0.0)
        torch.nn.init.constant_(self.W_bias[2].weight, 0.0)
        torch.nn.init.constant_(self.W_bias[0].bias, 0.0)
        torch.nn.init.constant_(self.W_bias[2].bias, 0.0)

    def forward(self, packed_input):
        x, speaker_embedding = packed_input
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        scale = self.W_scale(speaker_embedding)
        bias = self.W_bias(speaker_embedding)

        y = scale.unsqueeze(1) * ((x - mean) / var) + bias.unsqueeze(1)

        return y


class AdaIN1d(nn.Module):
    """
    MIT Licensed

    Copyright (c) 2022 Aaron (Yinghao) Li
    https://github.com/yl4579/StyleTTS/blob/main/models.py
    """

    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        s = torch.nn.functional.normalize(s)
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma.transpose(1, 2)) * self.norm(x.transpose(1, 2)).transpose(1, 2) + beta.transpose(1, 2)
