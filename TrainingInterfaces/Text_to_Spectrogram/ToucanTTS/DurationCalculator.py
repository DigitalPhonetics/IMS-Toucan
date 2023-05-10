# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Adapted by Florian Lux 2021

import matplotlib.pyplot as plt

import torch


class DurationCalculator(torch.nn.Module):

    def __init__(self, reduction_factor):
        self.reduction_factor = reduction_factor
        super().__init__()

    @torch.no_grad()
    def forward(self, att_ws, vis=None):
        """
        Convert alignment matrix to durations.
        """
        if vis is not None:
            plt.figure(figsize=(8, 4))
            plt.imshow(att_ws.cpu().numpy(), interpolation='nearest', aspect='auto', origin="lower")
            plt.xlabel("Inputs")
            plt.ylabel("Outputs")
            plt.tight_layout()
            plt.savefig(vis)
            plt.close()
        # calculate duration from 2d alignment matrix
        durations = torch.stack([att_ws.argmax(-1).eq(i).sum() for i in range(att_ws.shape[1])])
        return durations.view(-1) * self.reduction_factor
