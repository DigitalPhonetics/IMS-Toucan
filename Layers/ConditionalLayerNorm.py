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
                 normal_shape,
                 speaker_embedding_dim):
        super(ConditionalLayerNorm, self).__init__()
        if isinstance(normal_shape, int):
            self.normal_shape = normal_shape
        self.speaker_embedding_dim = speaker_embedding_dim
        self.W_scale = nn.Sequential(nn.Linear(self.speaker_embedding_dim, self.speaker_embedding_dim),
                                     nn.Tanh(),
                                     nn.Linear(self.speaker_embedding_dim, self.normal_shape),
                                     nn.Tanh(),
                                     nn.Linear(self.normal_shape, self.normal_shape))
        self.W_bias = nn.Sequential(nn.Linear(self.speaker_embedding_dim, self.speaker_embedding_dim),
                                    nn.Tanh(),
                                    nn.Linear(self.speaker_embedding_dim, self.normal_shape),
                                    nn.Tanh(),
                                    nn.Linear(self.normal_shape, self.normal_shape))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.W_scale[0].weight, 0.0)
        torch.nn.init.constant_(self.W_scale[2].weight, 0.0)
        torch.nn.init.constant_(self.W_scale[4].weight, 0.0)
        torch.nn.init.constant_(self.W_scale[0].bias, 1.0)
        torch.nn.init.constant_(self.W_scale[2].bias, 1.0)
        torch.nn.init.constant_(self.W_scale[4].bias, 1.0)
        torch.nn.init.constant_(self.W_bias[0].weight, 0.0)
        torch.nn.init.constant_(self.W_bias[2].weight, 0.0)
        torch.nn.init.constant_(self.W_bias[4].weight, 0.0)
        torch.nn.init.constant_(self.W_bias[0].bias, 0.0)
        torch.nn.init.constant_(self.W_bias[2].bias, 0.0)
        torch.nn.init.constant_(self.W_bias[4].bias, 0.0)

    def forward(self, x, speaker_embedding):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        scale = self.W_scale(speaker_embedding)
        bias = self.W_bias(speaker_embedding)

        y = scale.unsqueeze(1) * ((x - mean) / var) + bias.unsqueeze(1)

        return y


class SequentialWrappableConditionalLayerNorm(nn.Module):

    def __init__(self,
                 normal_shape,
                 speaker_embedding_dim):
        super(SequentialWrappableConditionalLayerNorm, self).__init__()
        if isinstance(normal_shape, int):
            self.normal_shape = normal_shape
        self.speaker_embedding_dim = speaker_embedding_dim
        self.W_scale = nn.Sequential(nn.Linear(self.speaker_embedding_dim, self.speaker_embedding_dim),
                                     nn.Tanh(),
                                     nn.Linear(self.speaker_embedding_dim, self.normal_shape),
                                     nn.Tanh(),
                                     nn.Linear(self.normal_shape, self.normal_shape))
        self.W_bias = nn.Sequential(nn.Linear(self.speaker_embedding_dim, self.speaker_embedding_dim),
                                    nn.Tanh(),
                                    nn.Linear(self.speaker_embedding_dim, self.normal_shape),
                                    nn.Tanh(),
                                    nn.Linear(self.normal_shape, self.normal_shape))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.W_scale[0].weight, 0.0)
        torch.nn.init.constant_(self.W_scale[2].weight, 0.0)
        torch.nn.init.constant_(self.W_scale[4].weight, 0.0)
        torch.nn.init.constant_(self.W_scale[0].bias, 1.0)
        torch.nn.init.constant_(self.W_scale[2].bias, 1.0)
        torch.nn.init.constant_(self.W_scale[4].bias, 1.0)
        torch.nn.init.constant_(self.W_bias[0].weight, 0.0)
        torch.nn.init.constant_(self.W_bias[2].weight, 0.0)
        torch.nn.init.constant_(self.W_bias[4].weight, 0.0)
        torch.nn.init.constant_(self.W_bias[0].bias, 0.0)
        torch.nn.init.constant_(self.W_bias[2].bias, 0.0)
        torch.nn.init.constant_(self.W_bias[4].bias, 0.0)

    def forward(self, packed_input):
        x, speaker_embedding = packed_input
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        scale = self.W_scale(speaker_embedding)
        bias = self.W_bias(speaker_embedding)

        y = scale.unsqueeze(1) * ((x - mean) / var) + bias.unsqueeze(1)

        return y
