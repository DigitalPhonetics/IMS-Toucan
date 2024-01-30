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
        self.l2_criterion = torch.nn.MSELoss(reduction="none")

    def forward(self, predicted_features, gold_features, features_lengths):
        """
        Args:
            predicted_features (Tensor): Batch of outputs before postnets (B, Lmax, odim).
            gold_features (Tensor): Batch of target features (B, Lmax, odim).
            features_lengths (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: L1 loss value.
        """

        # calculate losses
        distance_loss = self.l1_criterion(predicted_features, gold_features)

        # make weighted masks to ensure that long samples and short samples are all equally important
        out_masks = make_non_pad_mask(features_lengths).unsqueeze(-1).to(gold_features.device)
        out_weights = out_masks.float() / out_masks.sum(dim=1, keepdim=True).float()
        out_weights /= gold_features.size(0) * gold_features.size(-1)

        # apply weighted masks
        distance_loss = distance_loss.mul(out_weights).masked_select(out_masks).sum()

        return distance_loss
