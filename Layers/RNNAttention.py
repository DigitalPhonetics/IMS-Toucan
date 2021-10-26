# Published under Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Adapted by Florian Lux, 2021

import torch
import torch.nn.functional as F

from Utility.utils import make_pad_mask
from Utility.utils import to_device


def _apply_attention_constraint(e, last_attended_idx, backward_window=1, forward_window=3):
    """
    Apply monotonic attention constraint.

    This function apply the monotonic attention constraint
    introduced in `Deep Voice 3: Scaling
    Text-to-Speech with Convolutional Sequence Learning`_.

    Args:
        e (Tensor): Attention energy before applying softmax (1, T).
        last_attended_idx (int): The index of the inputs of the last attended [0, T].
        backward_window (int, optional): Backward window size in attention constraint.
        forward_window (int, optional): Forward window size in attetion constraint.

    Returns:
        Tensor: Monotonic constrained attention energy (1, T).

    .. _`Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning`:
        https://arxiv.org/abs/1710.07654
    """
    if e.size(0) != 1:
        raise NotImplementedError("Batch attention constraining is not yet supported.")
    backward_idx = last_attended_idx - backward_window
    forward_idx = last_attended_idx + forward_window
    if backward_idx > 0:
        e[:, :backward_idx] = -float("inf")
    if forward_idx < e.size(1):
        e[:, forward_idx:] = -float("inf")
    return e


class AttLoc(torch.nn.Module):
    """
    location-aware attention module.

    Reference: Attention-Based Models for Speech Recognition
        (https://arxiv.org/pdf/1506.07503.pdf)

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    :param bool han_mode: flag to switch on mode of hierarchical attention
        and not store pre_compute_enc_h
    """

    def __init__(self, eprojs, dunits, att_dim, aconv_chans, aconv_filts, han_mode=False):
        super(AttLoc, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.mlp_att = torch.nn.Linear(aconv_chans, att_dim, bias=False)
        self.loc_conv = torch.nn.Conv2d(1, aconv_chans, (1, 2 * aconv_filts + 1), padding=(0, aconv_filts), bias=False, )
        self.gvec = torch.nn.Linear(att_dim, 1)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None
        self.han_mode = han_mode

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self,
                enc_hs_pad,
                enc_hs_len,
                dec_z,
                att_prev,
                scaling=2.0,
                last_attended_idx=None,
                backward_window=1,
                forward_window=3):
        """
        Calculate AttLoc forward propagation.

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: previous attention weight (B x T_max)
        :param float scaling: scaling parameter before applying softmax
        :param torch.Tensor forward_window:
            forward window size when constraining attention
        :param int last_attended_idx: index of the inputs of the last attended
        :param int backward_window: backward window size in attention constraint
        :param int forward_window: forward window size in attention constraint
        :return: attention weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attention weights (B x T_max)
        :rtype: torch.Tensor
        """
        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None or self.han_mode:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)
        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        # initialize attention weight with uniform dist.
        if att_prev is None:
            # if no bias, 0 0-pad goes 0
            att_prev = 1.0 - make_pad_mask(enc_hs_len, device=dec_z.device).to(dtype=dec_z.dtype)
            att_prev = att_prev / att_prev.new(enc_hs_len).unsqueeze(-1)

        # att_prev: utt x frame -> utt x 1 x 1 x frame
        # -> utt x att_conv_chans x 1 x frame
        att_conv = self.loc_conv(att_prev.view(batch, 1, 1, self.h_length))
        # att_conv: utt x att_conv_chans x 1 x frame -> utt x frame x att_conv_chans
        att_conv = att_conv.squeeze(2).transpose(1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = self.mlp_att(att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(att_conv + self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE: consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_device(enc_hs_pad, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float("inf"))

        # apply monotonic attention constraint (mainly for TTS)
        if last_attended_idx is not None:
            e = _apply_attention_constraint(e, last_attended_idx, backward_window, forward_window)

        w = F.softmax(scaling * e, dim=1)

        # weighted sum over flames
        # utt x hdim
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)

        return c, w


class AttForwardTA(torch.nn.Module):
    """Forward attention with transition agent module.
    Reference:
    Forward attention in sequence-to-sequence acoustic modeling for speech synthesis
        (https://arxiv.org/pdf/1807.06736.pdf)
    :param int eunits: # units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    :param int odim: output dimension
    """

    def __init__(self, eunits, dunits, att_dim, aconv_chans, aconv_filts, odim):
        super(AttForwardTA, self).__init__()
        self.mlp_enc = torch.nn.Linear(eunits, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.mlp_ta = torch.nn.Linear(eunits + dunits + odim, 1)
        self.mlp_att = torch.nn.Linear(aconv_chans, att_dim, bias=False)
        self.loc_conv = torch.nn.Conv2d(1, aconv_chans, (1, 2 * aconv_filts + 1), padding=(0, aconv_filts), bias=False, )
        self.gvec = torch.nn.Linear(att_dim, 1)
        self.dunits = dunits
        self.eunits = eunits
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None
        self.trans_agent_prob = 0.5

    def reset(self):
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None
        self.trans_agent_prob = 0.5

    def forward(self,
                enc_hs_pad,
                enc_hs_len,
                dec_z,
                att_prev,
                out_prev,
                scaling=1.0,
                last_attended_idx=None,
                backward_window=1,
                forward_window=3):
        """
        Calculate AttForwardTA forward propagation.

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B, Tmax, eunits)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: decoder hidden state (B, dunits)
        :param torch.Tensor att_prev: attention weights of previous step
        :param torch.Tensor out_prev: decoder outputs of previous step (B, odim)
        :param float scaling: scaling parameter before applying softmax
        :param int last_attended_idx: index of the inputs of the last attended
        :param int backward_window: backward window size in attention constraint
        :param int forward_window: forward window size in attetion constraint
        :return: attention weighted encoder state (B, dunits)
        :rtype: torch.Tensor
        :return: previous attention weights (B, Tmax)
        :rtype: torch.Tensor
        """
        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        if att_prev is None:
            # initial attention will be [1, 0, 0, ...]
            att_prev = enc_hs_pad.new_zeros(*enc_hs_pad.size()[:2])
            att_prev[:, 0] = 1.0

        # att_prev: utt x frame -> utt x 1 x 1 x frame
        # -> utt x att_conv_chans x 1 x frame
        att_conv = self.loc_conv(att_prev.view(batch, 1, 1, self.h_length))
        # att_conv: utt x att_conv_chans x 1 x frame -> utt x frame x att_conv_chans
        att_conv = att_conv.squeeze(2).transpose(1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = self.mlp_att(att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(att_conv + self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_device(enc_hs_pad, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float("inf"))

        # apply monotonic attention constraint (mainly for TTS)
        if last_attended_idx is not None:
            e = _apply_attention_constraint(e, last_attended_idx, backward_window, forward_window)

        w = F.softmax(scaling * e, dim=1)

        # forward attention
        att_prev_shift = F.pad(att_prev, (1, 0))[:, :-1]
        w = (self.trans_agent_prob * att_prev + (1 - self.trans_agent_prob) * att_prev_shift) * w
        # NOTE: clamp is needed to avoid nan gradient
        w = F.normalize(torch.clamp(w, 1e-6), p=1, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)

        # update transition agent prob
        self.trans_agent_prob = torch.sigmoid(self.mlp_ta(torch.cat([c, out_prev, dec_z], dim=1)))

        return c, w
