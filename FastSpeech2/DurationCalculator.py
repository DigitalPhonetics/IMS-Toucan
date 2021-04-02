# -*- coding: utf-8 -*-

# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import matplotlib.pyplot as plt

import torch


class DurationCalculator(torch.nn.Module):
    """
    Duration calculator module.
    """

    def __init__(self, reduction_factor):
        """
        Initialize duration calculator.
        """
        self.reduction_factor = reduction_factor
        super().__init__()

    @torch.no_grad()
    def forward(self, att_ws: torch.Tensor, vis=None):
        """
        Convert attention weight to durations.
        Args:
            att_ws (Tensor): Attention weight tensor (L, T) or (#layers, #heads, L, T).
        Returns:
            LongTensor: Duration of each input (T,).
            Tensor: Focus rate value.
        """
        duration = self._calculate_duration(att_ws, vis=vis)
        focus_rate = self._calculate_focus_rate(att_ws)
        return duration, focus_rate

    @staticmethod
    def _calculate_focus_rate(att_ws):
        # transformer case -> (#layers, #heads, L, T)
        return att_ws.max(dim=-1)[0].mean(dim=-1).max()

    def _calculate_duration(self, att_ws, vis):
        # get the most diagonal head according to focus rate
        att_ws = torch.cat([att_w for att_w in att_ws], dim=0)  # (#heads * #layers, L, T)
        diagonal_scores = att_ws.max(dim=-1)[0].mean(dim=-1)  # (#heads * #layers,)
        diagonal_head_idx = diagonal_scores.argmax()
        att_ws = att_ws[diagonal_head_idx]  # (L, T)
        if vis is not None:
            plt.figure(figsize=(8, 4))
            plt.imshow(att_ws.cpu().numpy(), interpolation='nearest', aspect='auto', origin="lower")
            plt.xlabel("Inputs")
            plt.ylabel("Outputs")
            plt.tight_layout()
            plt.savefig(vis)
            plt.close()
        # calculate duration from 2d attention weight
        durations = torch.stack([att_ws.argmax(-1).eq(i).sum() for i in range(att_ws.shape[1])])
        return durations.view(-1) * self.reduction_factor
