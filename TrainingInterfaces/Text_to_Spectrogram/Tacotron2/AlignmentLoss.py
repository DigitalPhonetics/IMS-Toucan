import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import jit
from numba import prange


# Implementation by:
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/FastPitch/fastpitch/alignment.py
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/FastPitch/fastpitch/attn_loss_function.py


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

    def __init__(self, blank_logprob=-10):
        """
        The RAD-TTS Paper says the following about the blank_logprob:

        In practice, setting the blank emission probability blank_logprob to be
        roughly the value of the largest of the initial activations
        significantly improves convergence rates. The reasoning behind
        this is that it relaxes the monotonic constraint, allowing
        the objective function to construct paths while optionally
        skipping over some text tokens, notably ones that have not
        been sufficiently trained on during early iterations. As training
        proceeds, the probabilities of the skipped text token
        increases, despite the existence of the blank tokens, allowing us
        to extract clean monotonic alignments.

        -1 is given as default in the paper, but I find that the largest
        initial activations are more around -3 and there is large variance,
        so I decided to go for -10.
        """
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=3)
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)
        self.blank_logprob = blank_logprob

    def forward(self, attn_logprob, text_lens, spectrogram_lens):
        """
        Args:
            attn_logprob: batch x 1 x max(mel_lens) x max(text_lens) batched
            tensor of attention log probabilities, padded to length of longest
            sequence in each dimension
            text_lens: batch-D vector of length of each text sequence
            spectrogram_lens: batch-D vector of length of each mel sequence
        """
        # The CTC loss module assumes the existence of a blank token
        # that can be optionally inserted anywhere in the sequence for
        # a fixed probability.
        # A row must be added to the attention matrix to account for this
        attn_logprob_padded = F.pad(input=attn_logprob, pad=(1, 0, 0, 0, 0, 0, 0, 0), value=self.blank_logprob)

        # uncomment for figuring out the largest initial activation
        # print(torch.max(attn_logprob))

        total_loss = 0.0

        # for-loop over batch because of variable-length
        # sequences
        for bid in range(attn_logprob.shape[0]):
            # construct the target sequence. Every
            # text token is mapped to a unique sequence number,
            # thereby ensuring the monotonicity constraint
            target_seq = torch.arange(1, text_lens[bid] + 1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[: spectrogram_lens[bid], :, : text_lens[bid] + 1]

            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            loss = self.ctc_loss(curr_logprob,
                                 target_seq,
                                 input_lengths=spectrogram_lens[bid: bid + 1],
                                 target_lengths=text_lens[bid: bid + 1])
            total_loss = total_loss + loss
        # average cost over batch
        total_loss = total_loss / attn_logprob.shape[0]
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
    """
    mas with hardcoded width=1
    """
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

    def __init__(self,
                 bin_warmup_steps=10000,
                 bin_start_steps=60000,
                 forward_start_steps=10000,
                 include_forward_loss=False,  # something doesn't work right, causes stalling, disabled until figured out.
                 forward_loss_weight=0.01):
        super().__init__()
        if include_forward_loss:
            self.l_forward_func = ForwardSumLoss()
        self.include_forward_loss = include_forward_loss
        self.l_bin_func = BinLoss()
        self.bin_warmup_steps = bin_warmup_steps
        self.bin_start_steps = bin_start_steps
        self.forward_start_steps = forward_start_steps
        self.forward_loss_weight = forward_loss_weight

    def forward(self, soft_attention, in_lens, out_lens, step):

        soft_attention = soft_attention.unsqueeze(1)

        bin_weight = min(((step - self.bin_start_steps) / self.bin_warmup_steps) / 100, 0.01)

        if self.include_forward_loss and self.forward_start_steps < step:
            l_forward = self.l_forward_func(torch.log(soft_attention), in_lens, out_lens) * self.forward_loss_weight
        else:
            l_forward = 0.0

        if self.bin_start_steps < step:
            l_bin = bin_weight * self.l_bin_func(soft_attention, in_lens, out_lens)
        else:
            l_bin = 0.0

        return l_forward + l_bin
