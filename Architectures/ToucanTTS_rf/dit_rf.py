import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0.):
        super().__init__()
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.drop = nn.Dropout(p_dropout)
        self.act1 = nn.GELU(approximate="tanh")

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = self.act1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask

class MultiHeadAttention(nn.Module):
    def __init__(self, channels, out_channels, n_heads, p_dropout=0.):
        super().__init__()
        assert channels % n_heads == 0
        self.n_heads = n_heads
        self.k_channels = channels // n_heads

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)
        x = self.attention(q, k, v, mask=attn_mask)
        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        output = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=self.p_dropout if self.training else 0)
        return output.transpose(2, 3).contiguous().view(b, d, t_t)

class DiTConVBlock(nn.Module):
    def __init__(self, hidden_channels, out_channels, filter_channels, num_heads, kernel_size=3, p_dropout=0.1, gin_channels=0, gin_mu_channels=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_channels + out_channels, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadAttention(hidden_channels + out_channels, hidden_channels + out_channels, num_heads, p_dropout)
        self.norm2 = nn.LayerNorm(hidden_channels + out_channels, elementwise_affine=False, eps=1e-6)
        self.mlp = FFN(hidden_channels + out_channels, hidden_channels + out_channels, filter_channels, kernel_size, p_dropout=p_dropout)

        # Separate AdaLN for c and mu
        self.adaLN_modulation_c = nn.Sequential(
            nn.Linear(gin_channels, hidden_channels + out_channels) if gin_channels != hidden_channels + out_channels else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_channels + out_channels, 6 * (hidden_channels + out_channels), bias=True)
        )

        self.adaLN_modulation_mu = nn.Sequential(
            nn.Linear(gin_mu_channels, hidden_channels + out_channels) if gin_mu_channels != hidden_channels + out_channels else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_channels + out_channels, 6 * (hidden_channels + out_channels), bias=True)
        )

    def forward(self, x, c, mu, x_mask):
        """
        Args:
            x : [batch_size, channel, time]
            c : [batch_size, channel] (global conditioning)
            mu : [batch_size, channel] (separate global conditioning)
            x_mask : [batch_size, 1, time]
        return the same shape as x
        """
        x = x * x_mask
        attn_mask = x_mask.unsqueeze(1) * x_mask.unsqueeze(-1)  # shape: [batch_size, 1, time, time]

        # Compute modulation values for c and mu separately
        shift_msa_c, scale_msa_c, gate_msa_c, shift_mlp_c, scale_mlp_c, gate_mlp_c = self.adaLN_modulation_c(c).unsqueeze(2).chunk(6, dim=1)
        shift_msa_mu, scale_msa_mu, gate_msa_mu, shift_mlp_mu, scale_mlp_mu, gate_mlp_mu = self.adaLN_modulation_mu(mu).unsqueeze(2).chunk(6, dim=1)

        # Combine modulations from c and mu (sum them to allow independent influence)
        shift_msa = shift_msa_c + shift_msa_mu
        scale_msa = scale_msa_c + scale_msa_mu
        gate_msa = gate_msa_c + gate_msa_mu

        shift_mlp = shift_mlp_c + shift_mlp_mu
        scale_mlp = scale_mlp_c + scale_mlp_mu
        gate_mlp = gate_mlp_c + gate_mlp_mu

        x = x + gate_msa * self.attn(self.modulate(self.norm1(x.transpose(1, 2)).transpose(1, 2), shift_msa, scale_msa), attn_mask) * x_mask
        x = x + gate_mlp * self.mlp(self.modulate(self.norm2(x.transpose(1, 2)).transpose(1, 2), shift_mlp, scale_mlp), x_mask) * x_mask
        return x

    @staticmethod
    def modulate(x, shift, scale):
        return x * (1 + scale) + shift
