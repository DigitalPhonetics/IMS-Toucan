"""
Taken from ESPNet
Adapted by Flux
"""

import torch

from Utility.utils import make_non_pad_mask


class ToucanTTSLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.l1_criterion = torch.nn.L1Loss(reduction="none")

    def forward(self, after_outs, before_outs, ys, olens, ):
        """
        Args:
            after_outs (Tensor): Batch of outputs after postnets (B, Lmax, odim).
            before_outs (Tensor): Batch of outputs before postnets (B, Lmax, odim).
            ys (Tensor): Batch of target features (B, Lmax, odim).
            olens (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: L1 loss value.
        """

        # calculate loss
        l1_loss = self.l1_criterion(before_outs, ys)
        if after_outs is not None:
            l1_loss = l1_loss + self.l1_criterion(after_outs, ys)

        # make weighted mask and apply it
        out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
        out_masks = torch.nn.functional.pad(out_masks.transpose(1, 2), [0, ys.size(1) - out_masks.size(1), 0, 0, 0, 0], value=False).transpose(1, 2)
        out_weights = out_masks.float() / out_masks.sum(dim=1, keepdim=True).float()
        out_weights /= ys.size(0) * ys.size(2)

        # apply weight
        l1_loss = l1_loss.mul(out_weights).masked_select(out_masks).sum()

        return l1_loss
