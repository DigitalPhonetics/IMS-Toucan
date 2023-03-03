import torch
import torch.nn as nn
import torch.nn.functional as F

from Layers.ConditionalLayerNorm import ConditionalLayerNorm
from Layers.ConditionalLayerNorm import SequentialWrappableConditionalLayerNorm
from Layers.LayerNorm import LayerNorm
from TrainingInterfaces.Text_to_Spectrogram.PortaSpeech.wavenet import WN


class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def init_weights_func(m):
    classname = m.__class__.__name__
    if classname.find("Conv1d") != -1:
        torch.nn.init.xavier_uniform_(m.weight)


class ResidualBlock(nn.Module):
    """Implements conv->PReLU->norm n-times"""

    def __init__(self, channels, kernel_size, dilation, n=2, norm_type='bn', dropout=0.0,
                 c_multiple=2, ln_eps=1e-12, spk_emb_size=256):
        super(ResidualBlock, self).__init__()

        if norm_type == 'bn':
            norm_builder = lambda: nn.BatchNorm1d(channels)
        elif norm_type == 'in':
            norm_builder = lambda: nn.InstanceNorm1d(channels, affine=True)
        elif norm_type == 'gn':
            norm_builder = lambda: nn.GroupNorm(8, channels)
        elif norm_type == 'ln':
            norm_builder = lambda: LayerNorm(channels, dim=1, eps=ln_eps)
        elif norm_type == "cln":
            # condition sequence on an embedding vector while performing layer norm
            norm_builder = lambda: SequentialWrappableConditionalLayerNorm(normal_shape=1,
                                                                           speaker_embedding_dim=spk_emb_size)
        else:
            norm_builder = lambda: nn.Identity()

        self.blocks = [
            nn.Sequential(
                norm_builder(),
                nn.Conv1d(channels, c_multiple * channels, kernel_size, dilation=dilation,
                          padding=(dilation * (kernel_size - 1)) // 2),
                LambdaLayer(lambda x: x * kernel_size ** -0.5),
                nn.GELU(),
                nn.Conv1d(c_multiple * channels, channels, 1, dilation=dilation),
            )
            for i in range(n)
        ]

        self.blocks = nn.ModuleList(self.blocks)
        self.dropout = dropout
        self.norm_type = norm_type

    def forward(self, x_utt_emb):
        x = x_utt_emb[0]
        # have to upack the arguments from a single one because this is used inside a torch.nn.Sequential
        utt_emb = x_utt_emb[1]
        nonpadding = (x.abs().sum(1) > 0).float()[:, None, :]
        for b in self.blocks:
            if utt_emb is not None and self.norm_type == "cln":
                x_ = b((x, utt_emb))
            else:
                x_ = b(x)
            if self.dropout > 0 and self.training:
                x_ = F.dropout(x_, self.dropout, training=self.training)
            x = x + x_
            x = x * nonpadding
        return (x, utt_emb)


class ConvBlocks(nn.Module):
    """Decodes the expanded phoneme encoding into spectrograms"""

    def __init__(self, hidden_size, out_dims, dilations, kernel_size,
                 norm_type='ln', layers_in_block=2, c_multiple=2,
                 dropout=0.0, ln_eps=1e-5, spk_emb_size=256,
                 init_weights=True, is_BTC=True, num_layers=None, post_net_kernel=3):
        super(ConvBlocks, self).__init__()
        self.is_BTC = is_BTC
        if norm_type == "cln":
            self.utterance_conditioning_enabled = True
        else:
            self.utterance_conditioning_enabled = False
        if num_layers is not None:
            dilations = [1] * num_layers
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_size, kernel_size, d,
                            n=layers_in_block, norm_type=norm_type, c_multiple=c_multiple,
                            dropout=dropout, ln_eps=ln_eps, spk_emb_size=spk_emb_size)
              for d in dilations],
        )
        if norm_type == 'bn':
            norm = nn.BatchNorm1d(hidden_size)
        elif norm_type == 'in':
            norm = nn.InstanceNorm1d(hidden_size, affine=True)
        elif norm_type == 'gn':
            norm = nn.GroupNorm(8, hidden_size)
        elif norm_type == 'ln':
            norm = LayerNorm(hidden_size, dim=1, eps=ln_eps)
        elif norm_type == "cln":
            # condition sequence on an embedding vector while performing layer norm
            norm = ConditionalLayerNorm(normal_shape=1, speaker_embedding_dim=spk_emb_size)
        self.last_norm = norm
        self.post_net1 = nn.Conv1d(hidden_size, out_dims, kernel_size=post_net_kernel,
                                   padding=post_net_kernel // 2)
        if init_weights:
            self.apply(init_weights_func)

    def forward(self, x, nonpadding=None, utt_emb=None):
        """
        :param x: [B, T, H]
        :return:  [B, T, H]
        """
        if self.is_BTC:
            x = x.transpose(1, 2)
        if nonpadding is None:
            nonpadding = (x.abs().sum(1) > 0).float()[:, None, :]
        elif self.is_BTC:
            nonpadding = nonpadding.transpose(1, 2)
        x, _ = self.res_blocks((x, utt_emb))
        x = x * nonpadding
        if self.utterance_conditioning_enabled:
            x = self.last_norm(x, speaker_embedding=utt_emb) * nonpadding
        else:
            x = self.last_norm(x) * nonpadding
        x = self.post_net1(x) * nonpadding
        if self.is_BTC:
            x = x.transpose(1, 2)
        return x


class ConditionalConvBlocks(ConvBlocks):

    def __init__(self, hidden_size, c_cond, c_out, dilations, kernel_size,
                 norm_type='ln', layers_in_block=2, c_multiple=2,
                 dropout=0.0, ln_eps=1e-5, init_weights=True, is_BTC=True, num_layers=None,
                 spk_emb_size=256):
        super().__init__(hidden_size, c_out, dilations, kernel_size,
                         norm_type, layers_in_block, c_multiple,
                         dropout, ln_eps, spk_emb_size, init_weights, is_BTC=False, num_layers=num_layers)
        self.g_prenet = nn.Conv1d(c_cond, hidden_size, 3, padding=1)
        self.is_BTC_ = is_BTC
        if init_weights:
            self.g_prenet.apply(init_weights_func)

    def forward(self, x, cond, nonpadding=None, utt_emb=None):
        if self.is_BTC_:
            x = x.transpose(1, 2)
            cond = cond.transpose(1, 2)
            if nonpadding is not None:
                nonpadding = nonpadding.transpose(1, 2)
        if nonpadding is None:
            nonpadding = x.abs().sum(1)[:, None]
        x = x + self.g_prenet(cond)
        x = x * nonpadding
        x = super(ConditionalConvBlocks, self).forward(x, utt_emb=utt_emb)  # input needs to be BTC
        if self.is_BTC_:
            x = x.transpose(1, 2)
        return x


class FlipLayer(nn.Module):

    def forward(self, x, nonpadding, cond=None, reverse=False):
        x = torch.flip(x, [1])
        return x


class CouplingLayer(nn.Module):

    def __init__(self, c_in, hidden_size, kernel_size, n_layers, p_dropout=0, c_in_g=0, nn_type='wn'):
        super().__init__()
        self.channels = c_in
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.c_half = c_in // 2

        self.pre = nn.Conv1d(self.c_half, hidden_size, 1)
        if nn_type == 'wn':
            self.enc = WN(hidden_size, kernel_size, 1, n_layers, p_dropout=p_dropout, c_cond=c_in_g)
        elif nn_type == 'conv':
            self.enc = ConditionalConvBlocks(
                hidden_size, c_in_g, hidden_size, None, kernel_size,
                layers_in_block=1, is_BTC=False, num_layers=n_layers)
        self.post = nn.Conv1d(hidden_size, self.c_half, 1)

    def forward(self, x, nonpadding, cond=None, reverse=False):
        x0, x1 = x[:, :self.c_half], x[:, self.c_half:]
        x_ = self.pre(x0) * nonpadding
        x_ = self.enc(x_, nonpadding=nonpadding, cond=cond)
        m = self.post(x_)
        x1 = m + x1 if not reverse else x1 - m
        x = torch.cat([x0, x1], 1)
        return x * nonpadding


class ResFlow(nn.Module):

    def __init__(self,
                 c_in,
                 hidden_size,
                 kernel_size,
                 n_flow_layers,
                 n_flow_steps=22,
                 c_cond=0,
                 nn_type='wn'):
        super().__init__()
        self.flows = nn.ModuleList()
        for i in range(n_flow_steps):
            self.flows.append(CouplingLayer(c_in, hidden_size, kernel_size, n_flow_layers, c_in_g=c_cond, nn_type=nn_type))
            self.flows.append(FlipLayer())

    def forward(self, x, nonpadding, cond=None, reverse=False):
        for flow in (self.flows if not reverse else reversed(self.flows)):
            x = flow(x, nonpadding, cond=cond, reverse=reverse)
        return x
