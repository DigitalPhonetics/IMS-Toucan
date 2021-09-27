import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import jit
from numba import prange


# Implementation by:
# Lee, K. (2021). Comprehensive-Transformer-TTS (Version 0.1.1) [Computer software]. https://doi.org/10.5281/zenodo.5526991

# @misc{Lee_ComprehensiveTransformerTTS_2021,
# author = {Lee, Keon},
# doi = {10.5281/zenodo.5526991},
# month = {9},
# title = {{Comprehensive-Transformer-TTS}},
# url = {https://github.com/keonlee9420/Comprehensive-Transformer-TTS},
# year = {2021}
# }


# Concept by:
# https://arxiv.org/abs/2108.10447

# @misc{badlani2021tts,
#       title={One TTS Alignment To Rule Them All},
#       author={Rohan Badlani and Adrian Łancucki and Kevin J. Shih and Rafael Valle and Wei Ping and Bryan Catanzaro},
#       year={2021},
#       eprint={2108.10447},
#       archivePrefix={arXiv},
#       primaryClass={cs.SD}
# }


# Initial idea in:
# https://openreview.net/forum?id=0NQwnnwAORi

# @inproceedings{shih2021radtts,
#                title={{RAD}-{TTS}: Parallel Flow-Based {TTS} with Robust Alignment Learning and Diverse Synthesis},
#                author={Kevin J. Shih and Rafael Valle and Rohan Badlani and Adrian Lancucki and Wei Ping and Bryan Catanzaro},
#                booktitle={ICML Workshop on Invertible Neural Networks, Normalizing Flows, and Explicit Likelihood Models},
#                year={2021},
#                url={https://openreview.net/forum?id=0NQwnnwAORi}
# }

class ForwardSumLoss(nn.Module):

    def __init__(self, blank_logprob=-1):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=3)
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)
        self.blank_logprob = blank_logprob

    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = F.pad(input=attn_logprob, pad=(1, 0), value=self.blank_logprob)

        total_loss = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid] + 1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[: query_lens[bid], :, : key_lens[bid] + 1]

            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            loss = self.ctc_loss(
                curr_logprob,
                target_seq,
                input_lengths=query_lens[bid: bid + 1],
                target_lengths=key_lens[bid: bid + 1],
                )
            total_loss += loss

        total_loss /= attn_logprob.shape[0]
        return total_loss


class BinLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, soft_attention, in_lens, out_lens):
        hard_attention = binarize_attention_parallel(soft_attention, in_lens, out_lens)
        log_sum = torch.log(torch.clamp(soft_attention[hard_attention == 1], min=1e-12)).sum()
        return -log_sum / hard_attention.sum()


def binarize_attention_parallel(attn, in_lens, out_lens):
    """
    Binarizes attention with MAS.
    These will no longer receive a gradient.
    Args:
        attn: B x 1 x max_mel_len x max_text_len
    """
    with torch.no_grad():
        attn_cpu = attn.data.cpu().numpy()
        attn_out = b_mas(attn_cpu, in_lens.cpu().numpy(), out_lens.cpu().numpy(), width=1)
    return torch.from_numpy(attn_out).to(attn.device)


@jit(nopython=True, parallel=True)
def b_mas(b_attn_map, in_lens, out_lens, width=1):
    assert width == 1
    attn_out = np.zeros_like(b_attn_map)
    for b in prange(b_attn_map.shape[0]):
        out = mas_width1(b_attn_map[b, 0, : out_lens[b], : in_lens[b]])
        attn_out[b, 0, : out_lens[b], : in_lens[b]] = out
    return attn_out


@jit(nopython=True)
def mas_width1(attn_map):
    """mas with hardcoded width=1"""
    # assumes mel x text
    opt = np.zeros_like(attn_map)
    attn_map = np.log(attn_map)
    attn_map[0, 1:] = -np.inf
    log_p = np.zeros_like(attn_map)
    log_p[0, :] = attn_map[0, :]
    prev_ind = np.zeros_like(attn_map, dtype=np.int64)
    for i in range(1, attn_map.shape[0]):
        for j in range(attn_map.shape[1]):  # for each text dim
            prev_log = log_p[i - 1, j]
            prev_j = j

            if j - 1 >= 0 and log_p[i - 1, j - 1] >= log_p[i - 1, j]:
                prev_log = log_p[i - 1, j - 1]
                prev_j = j - 1

            log_p[i, j] = attn_map[i, j] + prev_log
            prev_ind[i, j] = prev_j

    # now backtrack
    curr_text_idx = attn_map.shape[1] - 1
    for i in range(attn_map.shape[0] - 1, -1, -1):
        opt[i, curr_text_idx] = 1
        curr_text_idx = prev_ind[i, curr_text_idx]
    opt[0, curr_text_idx] = 1
    return opt


class AlignmentLoss(nn.Module):
    """
    Combination according to paper with an added warmup phase directly in the loss
    """

    def __init__(self, bin_warmup_steps=10000, bin_start_steps=20000, include_forward_loss=False):
        super().__init__()
        if include_forward_loss:
            self.l_forward_func = ForwardSumLoss()
        self.include_forward_loss = include_forward_loss
        self.l_bin_func = BinLoss()
        self.bin_warmup_steps = bin_warmup_steps
        self.bin_start_steps = bin_start_steps

    def forward(self, soft_attention, in_lens, out_lens, step):

        soft_attention = soft_attention.unsqueeze(1)
        bin_weight = min(((step - self.bin_start_steps) / self.bin_warmup_steps) / 2, 0.5)

        if self.include_forward_loss:
            l_forward = self.l_forward_func(torch.log(soft_attention), in_lens, out_lens)
            # this is not the proper way to get log_probs, but the forward attention complicates things.
            # Luckily the forward attention does about the same as CTC, so it's not too necessary to have this.
        else:
            l_forward = 0.0

        if self.bin_start_steps < step:
            l_bin = bin_weight * self.l_bin_func(soft_attention, in_lens, out_lens)
        else:
            l_bin = 0.0

        return l_forward + l_bin
