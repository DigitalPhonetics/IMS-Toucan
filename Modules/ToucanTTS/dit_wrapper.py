"""
Copied from https://github.com/KdaiP/StableTTS by https://github.com/KdaiP

https://github.com/KdaiP/StableTTS/blob/eebb177ebf195fd1246dedabec4ef69d9351a4f8/models/estimator.py

Code is under MIT License
"""

import math

import torch
import torch.nn as nn

from Modules.ToucanTTS.dit import DiTConVBlock


class DitWrapper(nn.Module):
    """ add FiLM layer to condition time embedding to DiT """

    def __init__(self, hidden_channels, out_channels, filter_channels, num_heads, kernel_size=3, p_dropout=0.1, gin_channels=0, time_channels=0):
        super().__init__()
        if gin_channels is None:
            gin_channels = 0
        self.time_fusion = FiLMLayer(hidden_channels, out_channels, time_channels)
        self.conv1 = ConvNeXtBlock(hidden_channels, out_channels, filter_channels, gin_channels)
        self.conv2 = ConvNeXtBlock(hidden_channels, out_channels, filter_channels, gin_channels)
        self.conv3 = ConvNeXtBlock(hidden_channels, out_channels, filter_channels, gin_channels)
        self.block = DiTConVBlock(hidden_channels, out_channels, hidden_channels, num_heads, kernel_size, p_dropout, gin_channels)

    def forward(self, x, c, t, x_mask):
        x = self.time_fusion(x, t) * x_mask
        x = self.conv1(x, c, x_mask)
        x = self.conv2(x, c, x_mask)
        x = self.conv3(x, c, x_mask)
        x = self.block(x, c, x_mask)
        return x


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer
    Reference: https://arxiv.org/abs/1709.07871
    """

    def __init__(self, in_channels, out_channels, cond_channels):
        super(FiLMLayer, self).__init__()
        self.in_channels = in_channels
        self.film = nn.Conv1d(cond_channels, (in_channels + out_channels) * 2, 1)

    def forward(self, x, c):
        gamma, beta = torch.chunk(self.film(c.unsqueeze(2)), chunks=2, dim=1)
        return gamma * x + beta


class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels, gin_channels):
        super().__init__()
        self.dwconv = nn.Conv1d(in_channels + out_channels, in_channels + out_channels, kernel_size=7, padding=3, groups=in_channels + out_channels)
        self.norm = StyleAdaptiveLayerNorm(in_channels + out_channels, gin_channels)
        self.pwconv = nn.Sequential(nn.Linear(in_channels + out_channels, filter_channels),
                                    nn.GELU(),
                                    nn.Linear(filter_channels, in_channels + out_channels))

    def forward(self, x, c, x_mask) -> torch.Tensor:
        residual = x
        x = self.dwconv(x) * x_mask
        if c is not None:
            x = self.norm(x.transpose(1, 2), c)
        else:
            x = x.transpose(1, 2)
        x = self.pwconv(x).transpose(1, 2)
        x = residual + x
        return x * x_mask


class StyleAdaptiveLayerNorm(nn.Module):
    def __init__(self, in_channels, cond_channels):
        """
        Style Adaptive Layer Normalization (SALN) module.

        Parameters:
        in_channels: The number of channels in the input feature maps.
        cond_channels: The number of channels in the conditioning input.
        """
        super(StyleAdaptiveLayerNorm, self).__init__()
        self.in_channels = in_channels

        self.saln = nn.Linear(cond_channels, in_channels * 2, 1)
        self.norm = nn.LayerNorm(in_channels, elementwise_affine=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.saln.bias.data[:self.in_channels], 1)
        nn.init.constant_(self.saln.bias.data[self.in_channels:], 0)

    def forward(self, x, c):
        gamma, beta = torch.chunk(self.saln(c.unsqueeze(1)), chunks=2, dim=-1)
        return gamma * self.norm(x) + beta


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(in_channels, filter_channels),
            nn.SiLU(inplace=True),
            nn.Linear(filter_channels, out_channels)
        )

    def forward(self, x):
        return self.layer(x)


# reference: https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/components/decoder.py
class Decoder(nn.Module):
    def __init__(self, hidden_channels, out_channels, filter_channels, dropout=0.05, n_layers=1, n_heads=4, kernel_size=3, gin_channels=0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels

        self.time_embeddings = SinusoidalPosEmb(hidden_channels)
        self.time_mlp = TimestepEmbedding(hidden_channels, hidden_channels, filter_channels)

        self.blocks = nn.ModuleList([DitWrapper(hidden_channels, out_channels, filter_channels, n_heads, kernel_size, dropout, gin_channels, hidden_channels) for _ in range(n_layers)])
        self.final_proj = nn.Conv1d(hidden_channels + out_channels, out_channels, 1)

        self.initialize_weights()

    def initialize_weights(self):
        for block in self.blocks:
            nn.init.constant_(block.block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.block.adaLN_modulation[-1].bias, 0)

    def forward(self, x, mask, mu, t, c):
        """Forward pass of the UNet1DConditional model.

        Args:
            x (torch.Tensor): shape (batch_size, in_channels, time)
            mask (_type_): shape (batch_size, 1, time)
            t (_type_): shape (batch_size)
            c (_type_): shape (batch_size, gin_channels)

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        t = self.time_mlp(self.time_embeddings(t))

        x = torch.cat((x, mu), dim=1)

        for block in self.blocks:
            x = block(x, c, t, mask)

        output = self.final_proj(x * mask)

        return output * mask
