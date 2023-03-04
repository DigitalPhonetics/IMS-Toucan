"""
Taken from ESPNet
Adapted by Flux
"""

import torch

from Layers.DurationPredictor import DurationPredictorLoss
from Utility.utils import make_non_pad_mask


def weights_nonzero_speech(target):
    # target : B x T x mel
    # Assign weight 1.0 to all labels except for padding (id=0).
    dim = target.size(-1)
    return target.abs().sum(-1, keepdim=True).ne(0).float().repeat(1, 1, dim)


class ToucanTTSLoss(torch.nn.Module):

    def __init__(self):
        """
            use_masking (bool):
                Whether to apply masking for padded part in loss calculation.
            use_weighted_masking (bool):
                Whether to weighted masking in loss calculation.
        """
        super().__init__()

        # define criterions
        self.l1_criterion = torch.nn.L1Loss(reduction="none")
        self.mse_criterion = torch.nn.MSELoss(reduction="none")
        self.duration_criterion = DurationPredictorLoss(reduction="none")

    def forward(self, after_outs, before_outs, d_outs, p_outs, e_outs, ys,
                ds, ps, es, ilens, olens, ):
        """
        Args:
            after_outs (Tensor): Batch of outputs after postnets (B, Lmax, odim).
            before_outs (Tensor): Batch of outputs before postnets (B, Lmax, odim).
            d_outs (LongTensor): Batch of outputs of duration predictor (B, Tmax).
            p_outs (Tensor): Batch of outputs of pitch predictor (B, Tmax, 1).
            e_outs (Tensor): Batch of outputs of energy predictor (B, Tmax, 1).
            ys (Tensor): Batch of target features (B, Lmax, odim).
            ds (LongTensor): Batch of durations (B, Tmax).
            ps (Tensor): Batch of target token-averaged pitch (B, Tmax, 1).
            es (Tensor): Batch of target token-averaged energy (B, Tmax, 1).
            ilens (LongTensor): Batch of the lengths of each input (B,).
            olens (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: L1 loss value.
            Tensor: Duration predictor loss value.
            Tensor: Pitch predictor loss value.
            Tensor: Energy predictor loss value.

        """

        # calculate loss
        l1_loss = self.l1_criterion(before_outs, ys)
        if after_outs is not None:
            l1_loss = l1_loss + self.l1_criterion(after_outs, ys)
        duration_loss = self.duration_criterion(d_outs, ds)
        pitch_loss = self.mse_criterion(p_outs, ps)
        energy_loss = self.mse_criterion(e_outs, es)

        # make weighted mask and apply it
        out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
        out_masks = torch.nn.functional.pad(out_masks.transpose(1, 2),
                                            [0, ys.size(1) - out_masks.size(1), 0, 0, 0, 0], value=False).transpose(1, 2)

        out_weights = out_masks.float() / out_masks.sum(dim=1, keepdim=True).float()
        out_weights /= ys.size(0) * ys.size(2)
        duration_masks = make_non_pad_mask(ilens).to(ys.device)
        duration_weights = (duration_masks.float() / duration_masks.sum(dim=1, keepdim=True).float())
        duration_weights /= ds.size(0)

        # apply weight
        l1_loss = l1_loss.mul(out_weights).masked_select(out_masks).sum()
        duration_loss = (duration_loss.mul(duration_weights).masked_select(duration_masks).sum())
        pitch_masks = duration_masks.unsqueeze(-1)
        pitch_weights = duration_weights.unsqueeze(-1)
        pitch_loss = pitch_loss.mul(pitch_weights).masked_select(pitch_masks).sum()
        energy_loss = (energy_loss.mul(pitch_weights).masked_select(pitch_masks).sum())

        return l1_loss, duration_loss, pitch_loss, energy_loss
