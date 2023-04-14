import numpy as np
import scipy
import torch
import torch.distributions as dist
from torch import nn
from torch.nn import functional as F

from TrainingInterfaces.Text_to_Spectrogram.ToucanTTS import glow_utils
from TrainingInterfaces.Text_to_Spectrogram.ToucanTTS.wavenet import WN


class ActNorm(nn.Module):

    def __init__(self, channels, ddi=False, **kwargs):
        super().__init__()
        self.channels = channels
        self.initialized = not ddi

        self.logs = nn.Parameter(torch.zeros(1, channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x, x_mask=None, reverse=False, **kwargs):
        if x_mask is None:
            x_mask = torch.ones(x.size(0), 1, x.size(2)).to(device=x.device, dtype=x.dtype)
        x_len = torch.sum(x_mask, [1, 2])
        if not self.initialized:
            self.initialize(x, x_mask)
            self.initialized = True

        if reverse:
            z = (x - self.bias) * torch.exp(-self.logs) * x_mask
            logdet = torch.sum(-self.logs) * x_len
        else:
            z = (self.bias + torch.exp(self.logs) * x) * x_mask
            logdet = torch.sum(self.logs) * x_len  # [b]
        return z, logdet

    def store_inverse(self):
        pass

    def set_ddi(self, ddi):
        self.initialized = not ddi

    def initialize(self, x, x_mask):
        with torch.no_grad():
            denom = torch.sum(x_mask, [0, 2])
            m = torch.sum(x * x_mask, [0, 2]) / denom
            m_sq = torch.sum(x * x * x_mask, [0, 2]) / denom
            v = m_sq - (m ** 2)
            logs = 0.5 * torch.log(torch.clamp_min(v, 1e-6))

            bias_init = (-m * torch.exp(-logs)).view(*self.bias.shape).to(dtype=self.bias.dtype)
            logs_init = (-logs).view(*self.logs.shape).to(dtype=self.logs.dtype)

            self.bias.data.copy_(bias_init)
            self.logs.data.copy_(logs_init)


class InvConvNear(nn.Module):

    def __init__(self, channels, n_split=4, no_jacobian=False, lu=True, n_sqz=2, **kwargs):
        super().__init__()
        assert (n_split % 2 == 0)
        self.channels = channels
        self.n_split = n_split
        self.n_sqz = n_sqz
        self.no_jacobian = no_jacobian

        w_init = torch.linalg.qr(torch.FloatTensor(self.n_split, self.n_split).normal_(), 'complete')[0]
        if torch.det(w_init) < 0:
            w_init[:, 0] = -1 * w_init[:, 0]
        self.lu = lu
        if lu:
            # LU decomposition can slightly speed up the inverse
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_init.shape, dtype=float), -1)
            eye = np.eye(*w_init.shape, dtype=float)

            self.register_buffer('p', torch.Tensor(np_p.astype(float)))
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(float)))
            self.l = nn.Parameter(torch.Tensor(np_l.astype(float)), requires_grad=True)
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(float)), requires_grad=True)
            self.u = nn.Parameter(torch.Tensor(np_u.astype(float)), requires_grad=True)
            self.register_buffer('l_mask', torch.Tensor(l_mask))
            self.register_buffer('eye', torch.Tensor(eye))
        else:
            self.weight = nn.Parameter(w_init)

    def forward(self, x, x_mask=None, reverse=False, **kwargs):
        b, c, t = x.size()
        assert (c % self.n_split == 0)
        if x_mask is None:
            x_mask = 1
            x_len = torch.ones((b,), dtype=x.dtype, device=x.device) * t
        else:
            x_len = torch.sum(x_mask, [1, 2])

        x = x.view(b, self.n_sqz, c // self.n_split, self.n_split // self.n_sqz, t)
        x = x.permute(0, 1, 3, 2, 4).contiguous().view(b, self.n_split, c // self.n_split, t)

        if self.lu:
            self.weight, log_s = self._get_weight()
            logdet = log_s.sum()
            logdet = logdet * (c / self.n_split) * x_len
        else:
            logdet = torch.logdet(self.weight) * (c / self.n_split) * x_len  # [b]

        if reverse:
            if hasattr(self, "weight_inv"):
                weight = self.weight_inv
            else:
                weight = torch.inverse(self.weight.float()).to(dtype=self.weight.dtype)
            logdet = -logdet
        else:
            weight = self.weight
            if self.no_jacobian:
                logdet = 0

        weight = weight.view(self.n_split, self.n_split, 1, 1).to(x.device)
        z = F.conv2d(x, weight)

        z = z.view(b, self.n_sqz, self.n_split // self.n_sqz, c // self.n_split, t)
        z = z.permute(0, 1, 3, 2, 4).contiguous().view(b, c, t) * x_mask
        return z, logdet

    def _get_weight(self):
        l, log_s, u = self.l, self.log_s, self.u
        l = l * self.l_mask + self.eye
        u = u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(log_s))
        weight = torch.matmul(self.p, torch.matmul(l, u))
        return weight, log_s

    def store_inverse(self):
        weight, _ = self._get_weight()
        self.weight_inv = torch.inverse(weight.float()).to(next(self.parameters()).device)


class InvConv(nn.Module):

    def __init__(self, channels, no_jacobian=False, lu=True, **kwargs):
        super().__init__()
        w_shape = [channels, channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(float)
        LU_decomposed = lu
        if not LU_decomposed:
            # Sample a random orthogonal matrix:
            self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        else:
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_shape, dtype=float), -1)
            eye = np.eye(*w_shape, dtype=float)

            self.register_buffer('p', torch.Tensor(np_p.astype(float)))
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(float)))
            self.l = nn.Parameter(torch.Tensor(np_l.astype(float)))
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(float)))
            self.u = nn.Parameter(torch.Tensor(np_u.astype(float)))
            self.l_mask = torch.Tensor(l_mask)
            self.eye = torch.Tensor(eye)
        self.w_shape = w_shape
        self.LU = LU_decomposed
        self.weight = None

    def get_weight(self, device, reverse):
        w_shape = self.w_shape
        self.p = self.p.to(device)
        self.sign_s = self.sign_s.to(device)
        self.l_mask = self.l_mask.to(device)
        self.eye = self.eye.to(device)
        l = self.l * self.l_mask + self.eye
        u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
        dlogdet = self.log_s.sum()
        if not reverse:
            w = torch.matmul(self.p, torch.matmul(l, u))
        else:
            l = torch.inverse(l.double()).float()
            u = torch.inverse(u.double()).float()
            w = torch.matmul(u, torch.matmul(l, self.p.inverse()))
        return w.view(w_shape[0], w_shape[1], 1), dlogdet

    def forward(self, x, x_mask=None, reverse=False, **kwargs):
        """
        log-det = log|abs(|W|)| * pixels
        """
        b, c, t = x.size()
        if x_mask is None:
            x_len = torch.ones((b,), dtype=x.dtype, device=x.device) * t
        else:
            x_len = torch.sum(x_mask, [1, 2])
        logdet = 0
        if not reverse:
            weight, dlogdet = self.get_weight(x.device, reverse)
            z = F.conv1d(x, weight)
            if logdet is not None:
                logdet = logdet + dlogdet * x_len
            return z, logdet
        else:
            if self.weight is None:
                weight, dlogdet = self.get_weight(x.device, reverse)
            else:
                weight, dlogdet = self.weight, self.dlogdet
            z = F.conv1d(x, weight)
            if logdet is not None:
                logdet = logdet - dlogdet * x_len
            return z, logdet

    def store_inverse(self):
        self.weight, self.dlogdet = self.get_weight('cuda', reverse=True)


class CouplingBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, kernel_size, dilation_rate, n_layers,
                 gin_channels=0, p_dropout=0., sigmoid_scale=False, wn=None, use_weightnorm=False):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout
        self.sigmoid_scale = sigmoid_scale

        start = torch.nn.Conv1d(in_channels // 2, hidden_channels, 1)
        if use_weightnorm:
            start = torch.nn.utils.weight_norm(start)
        self.start = start
        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end = torch.nn.Conv1d(hidden_channels, in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end
        self.wn = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels, p_dropout, use_weightnorm=use_weightnorm)
        if wn is not None:
            self.wn.in_layers = wn.in_layers
            self.wn.res_skip_layers = wn.res_skip_layers

    def forward(self, x, x_mask=None, reverse=False, g=None, **kwargs):
        if x_mask is None:
            x_mask = 1
        x_0, x_1 = x[:, :self.in_channels // 2], x[:, self.in_channels // 2:]

        x = self.start(x_0) * x_mask
        x = self.wn(x, x_mask, g)
        out = self.end(x)

        z_0 = x_0
        m = out[:, :self.in_channels // 2, :]
        logs = out[:, self.in_channels // 2:, :]
        if self.sigmoid_scale:
            logs = torch.log(1e-6 + torch.sigmoid(logs + 2))
        if reverse:
            z_1 = (x_1 - m) * torch.exp(-logs) * x_mask
            logdet = torch.sum(-logs * x_mask, [1, 2])
        else:
            z_1 = (m + torch.exp(logs) * x_1) * x_mask
            logdet = torch.sum(logs * x_mask, [1, 2])
        z = torch.cat([z_0, z_1], 1)
        return z, logdet

    def store_inverse(self):
        self.wn.remove_weight_norm()


class Glow(nn.Module):

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_blocks,
                 n_layers,
                 condition_integration_projection,
                 p_dropout=0.,
                 n_split=4,
                 n_sqz=2,
                 sigmoid_scale=False,
                 text_condition_channels=0,
                 inv_conv_type='near',
                 share_cond_layers=False,
                 share_wn_layers=0,
                 use_weightnorm=True  # If weightnorm is set to false, we can deepcopy the module, which we need to be able to do to perform SWA. Without weightnorm, the module will probably take a little longer to converge.
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        self.n_split = n_split
        self.n_sqz = n_sqz
        self.sigmoid_scale = sigmoid_scale
        self.text_condition_channels = text_condition_channels
        self.share_cond_layers = share_cond_layers
        self.prior_dist = dist.Normal(0, 1)
        self.g_proj = condition_integration_projection
        if text_condition_channels != 0 and share_cond_layers:
            cond_layer = torch.nn.Conv1d(text_condition_channels * n_sqz, 2 * hidden_channels * n_layers, 1)
            if use_weightnorm:
                self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')
            else:
                self.cond_layer = cond_layer
        wn = None
        self.flows = nn.ModuleList()
        for b in range(n_blocks):
            self.flows.append(ActNorm(channels=in_channels * n_sqz))
            if inv_conv_type == 'near':
                self.flows.append(InvConvNear(channels=in_channels * n_sqz, n_split=n_split, n_sqz=n_sqz))
            if inv_conv_type == 'invconv':
                self.flows.append(InvConv(channels=in_channels * n_sqz))
            if share_wn_layers > 0:
                if b % share_wn_layers == 0:
                    wn = WN(hidden_channels, kernel_size, dilation_rate, n_layers, text_condition_channels * n_sqz, p_dropout, share_cond_layers, use_weightnorm=use_weightnorm)
            self.flows.append(
                CouplingBlock(
                    in_channels * n_sqz,
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate,
                    n_layers=n_layers,
                    gin_channels=text_condition_channels * n_sqz,
                    p_dropout=p_dropout,
                    sigmoid_scale=sigmoid_scale,
                    wn=wn,
                    use_weightnorm=use_weightnorm
                ))

    def forward(self, tgt_mels, infer, mel_out, encoded_texts, tgt_nonpadding):
        x_recon = mel_out.transpose(1, 2)
        g = x_recon
        B, _, T = g.shape
        if encoded_texts is not None and self.text_condition_channels != 0:
            g = torch.cat([g, encoded_texts.transpose(1, 2)], 1)
            g = self.g_proj(g)
        prior_dist = self.prior_dist
        if not infer:
            y_lengths = tgt_nonpadding.sum(-1)
            tgt_mels = tgt_mels.transpose(1, 2)
            z_postflow, ldj = self._forward(tgt_mels, tgt_nonpadding, g=g)
            ldj = ldj / y_lengths / 80
            try:
                postflow_loss = -prior_dist.log_prob(z_postflow).mean() - ldj.mean()
            except ValueError:
                print("log probability of plostflow could not be calculated for this step")
                postflow_loss = None
            return postflow_loss
        else:
            nonpadding = torch.ones_like(x_recon[:, :1, :]) if tgt_nonpadding is None else tgt_nonpadding
            z_post = torch.randn(x_recon.shape).to(g.device) * 0.8
            x_recon, _ = self._forward(z_post, nonpadding, g, reverse=True)
            return x_recon.transpose(1, 2)

    def _forward(self, x, x_mask=None, g=None, reverse=False, return_hiddens=False):
        logdet_tot = 0
        if not reverse:
            flows = self.flows
        else:
            flows = reversed(self.flows)
        if return_hiddens:
            hs = []
        if self.n_sqz > 1:
            x, x_mask_ = glow_utils.squeeze(x, x_mask, self.n_sqz)
            if g is not None:
                g, _ = glow_utils.squeeze(g, x_mask, self.n_sqz)
            x_mask = x_mask_
        if self.share_cond_layers and g is not None:
            g = self.cond_layer(g)
        for f in flows:
            x, logdet = f(x, x_mask, g=g, reverse=reverse)
            if return_hiddens:
                hs.append(x)
            logdet_tot += logdet
        if self.n_sqz > 1:
            x, x_mask = glow_utils.unsqueeze(x, x_mask, self.n_sqz)
        if return_hiddens:
            return x, logdet_tot, hs
        return x, logdet_tot

    def store_inverse(self):
        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(remove_weight_norm)
        for f in self.flows:
            f.store_inverse()
