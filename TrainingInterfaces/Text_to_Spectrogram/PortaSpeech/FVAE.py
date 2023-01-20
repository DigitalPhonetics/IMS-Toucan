import numpy as np
import torch
import torch.distributions as dist
from torch import nn

from TrainingInterfaces.Text_to_Spectrogram.PortaSpeech.PortaSpeechLayers import ConditionalConvBlocks
from TrainingInterfaces.Text_to_Spectrogram.PortaSpeech.PortaSpeechLayers import ResFlow


class FVAEEncoder(nn.Module):

    def __init__(self, c_in, hidden_size, c_latent, kernel_size,
                 n_layers, c_cond=0, p_dropout=0, strides=[4], norm_type="ln", spk_emb_size=256):
        super().__init__()
        self.strides = strides
        self.hidden_size = hidden_size
        if np.prod(strides) == 1:
            self.pre_net = nn.Conv1d(c_in, hidden_size, kernel_size=1)
        else:
            self.pre_net = nn.Sequential(
                *[nn.Conv1d(c_in, hidden_size, kernel_size=s * 2, stride=s, padding=s // 2) if i == 0 else
                  nn.Conv1d(hidden_size, hidden_size, kernel_size=s * 2, stride=s, padding=s // 2) for i, s in
                  enumerate(strides)])
        self.nn = ConditionalConvBlocks(hidden_size, c_cond, hidden_size, None, kernel_size, layers_in_block=2,
                                        is_BTC=False, num_layers=n_layers, norm_type=norm_type,
                                        spk_emb_size=spk_emb_size)
        self.out_proj = nn.Conv1d(hidden_size, c_latent * 2, 1)
        self.latent_channels = c_latent

    def forward(self, x, nonpadding, cond, utt_emb=None):
        x = self.pre_net(x)
        nonpadding = nonpadding[:, :, ::np.prod(self.strides)][:, :, :x.shape[-1]]
        x = x * nonpadding
        x = self.nn(x, nonpadding=nonpadding, cond=cond, utt_emb=utt_emb) * nonpadding
        x = self.out_proj(x)
        m, logs = torch.split(x, self.latent_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs))
        return z, m, logs, nonpadding


class FVAEDecoder(nn.Module):

    def __init__(self, c_latent, hidden_size, out_channels, kernel_size,
                 n_layers, c_cond=0, p_dropout=0, strides=[4], norm_type="ln", spk_emb_size=256):
        super().__init__()
        self.strides = strides
        self.hidden_size = hidden_size
        self.pre_net = nn.Sequential(*[nn.ConvTranspose1d(c_latent, hidden_size, kernel_size=s, stride=s) if i == 0 else
                                       nn.ConvTranspose1d(hidden_size, hidden_size, kernel_size=s, stride=s) for i, s in
                                       enumerate(strides)])
        self.nn = ConditionalConvBlocks(hidden_size, c_cond, hidden_size, [1] * n_layers, kernel_size,
                                        layers_in_block=2, is_BTC=False, norm_type=norm_type, spk_emb_size=spk_emb_size)
        self.out_proj = nn.Conv1d(hidden_size, out_channels, 1)

    def forward(self, x, nonpadding, cond, utt_emb=None):
        x = self.pre_net(x)
        if nonpadding is None:
            mask_padding = 1
        else:
            mask_padding = nonpadding
        x = x * mask_padding
        x = self.nn(x, nonpadding=nonpadding, cond=cond, utt_emb=utt_emb) * mask_padding
        x = self.out_proj(x)
        return x


class FVAE(nn.Module):

    def __init__(self,
                 c_in, c_out, hidden_size, c_latent,
                 kernel_size, enc_n_layers, dec_n_layers, c_cond, strides,
                 use_prior_flow, flow_hidden=None, flow_kernel_size=None, flow_n_steps=None,
                 norm_type="ln", spk_emb_size=256):
        super(FVAE, self).__init__()
        self.strides = strides
        self.hidden_size = hidden_size
        self.latent_size = c_latent
        self.use_prior_flow = use_prior_flow
        if np.prod(strides) == 1:
            self.g_pre_net = nn.Conv1d(c_cond, c_cond, kernel_size=1)
        else:
            self.g_pre_net = nn.Sequential(
                *[nn.Conv1d(c_cond, c_cond, kernel_size=s * 2, stride=s, padding=s // 2) for i, s in
                  enumerate(strides)])
        self.encoder = FVAEEncoder(c_in, hidden_size, c_latent, kernel_size, enc_n_layers, c_cond, strides=strides,
                                   norm_type=norm_type, spk_emb_size=spk_emb_size)
        if use_prior_flow:
            self.prior_flow = ResFlow(c_latent, flow_hidden, flow_kernel_size, flow_n_steps, 4, c_cond=c_cond)
        self.decoder = FVAEDecoder(c_latent, hidden_size, c_out, kernel_size, dec_n_layers, c_cond, strides=strides,
                                   norm_type=norm_type, spk_emb_size=spk_emb_size)
        self.prior_dist = dist.Normal(0, 1)

    def forward(self, x=None, nonpadding=None, cond=None, infer=False, noise_scale=1.0, utt_emb=None):
        """
        :param x: [B, C_in_out, T]
        :param nonpadding: [B, 1, T]
        :param cond: [B, C_g, T]
        :return:
        """
        if nonpadding is None:
            nonpadding = 1
        cond_sqz = self.g_pre_net(cond)
        if not infer:
            z_q, m_q, logs_q, nonpadding_sqz = self.encoder(x.transpose(1, 2), nonpadding, cond_sqz, utt_emb=utt_emb)
            q_dist = dist.Normal(torch.clamp(m_q, min=None, max=1e10), torch.clamp(logs_q, min=None, max=500).exp())
            if self.use_prior_flow:
                logqx = q_dist.log_prob(z_q)
                z_p = self.prior_flow(z_q, nonpadding_sqz, cond_sqz)
                # we have to be careful, because this can sometimes produce numbers <= 0,
                # or NaNs, which leads to an undefined log, which in turn triggers an error
                logpx = self.prior_dist.log_prob(z_p)
                loss_kl = ((logqx - logpx) * nonpadding_sqz).sum() / nonpadding_sqz.sum() / logqx.shape[1]
            else:
                loss_kl = torch.distributions.kl_divergence(q_dist, self.prior_dist)
                loss_kl = (loss_kl * nonpadding_sqz).sum() / nonpadding_sqz.sum() / z_q.shape[1]
                z_p = None
            return z_q, loss_kl, z_p, m_q, logs_q
        else:
            latent_shape = [cond_sqz.shape[0], self.latent_size, cond_sqz.shape[2]]
            z_p = torch.randn(latent_shape).to(cond.device) * noise_scale
            if self.use_prior_flow:
                z_p = self.prior_flow(z_p, 1, cond_sqz, reverse=True)
            return z_p
