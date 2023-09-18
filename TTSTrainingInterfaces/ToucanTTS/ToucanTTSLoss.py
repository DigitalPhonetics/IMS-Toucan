"""
Taken from ESPNet
Adapted by Flux
"""

import torch

from Layers.DurationPredictor import DurationPredictorLoss
from Utility.utils import make_non_pad_mask


class ToucanTTSLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.l1_criterion = torch.nn.L1Loss(reduction="none")
        self.l2_criterion = torch.nn.MSELoss(reduction="none")
        self.duration_criterion = DurationPredictorLoss(reduction="none")

    def forward(self, predicted_features, gold_features, features_lengths, text_lengths, gold_durations, predicted_durations, predicted_pitch, predicted_energy, gold_pitch, gold_energy):
        """
        Args:
            predicted_features (Tensor): Batch of outputs before postnets (B, Lmax, odim).
            gold_features (Tensor): Batch of target features (B, Lmax, odim).
            features_lengths (LongTensor): Batch of the lengths of each target (B,).
            gold_durations (LongTensor): Batch of durations (B, Tmax).
            gold_pitch (LongTensor): Batch of pitch (B, Tmax).
            gold_energy (LongTensor): Batch of energy (B, Tmax).
            predicted_durations (LongTensor): Batch of outputs of duration predictor (B, Tmax).
            predicted_pitch (LongTensor): Batch of outputs of pitch predictor (B, Tmax).
            predicted_energy (LongTensor): Batch of outputs of energy predictor (B, Tmax).
            text_lengths (LongTensor): Batch of the lengths of each input (B,).

        Returns:
            Tensor: L1 loss value.
            Tensor: Duration loss value
        """

        # calculate loss
        distance_loss = self.l1_criterion(predicted_features, gold_features)

        duration_loss = self.duration_criterion(predicted_durations, gold_durations)
        pitch_loss = self.l2_criterion(predicted_pitch, gold_pitch)
        energy_loss = self.l2_criterion(predicted_energy, gold_energy)

        # make weighted mask and apply it
        out_masks = make_non_pad_mask(features_lengths).unsqueeze(-1).to(gold_features.device)
        out_weights = out_masks.float() / out_masks.sum(dim=1, keepdim=True).float()
        out_weights /= gold_features.size(0) * gold_features.size(-1)  # make sure that long samples and short samples are all equally important
        duration_masks = make_non_pad_mask(text_lengths).to(gold_features.device)
        duration_weights = (duration_masks.float() / duration_masks.sum(dim=1, keepdim=True).float())
        variance_masks = duration_masks.unsqueeze(-1)
        variance_weights = duration_weights.unsqueeze(-1)
        pitch_loss = pitch_loss.mul(variance_weights).masked_select(variance_masks).sum()
        energy_loss = (energy_loss.mul(variance_weights).masked_select(variance_masks).sum())

        # apply weight
        distance_loss = distance_loss.mul(out_weights).masked_select(out_masks).sum()
        duration_loss = (duration_loss.mul(duration_weights).masked_select(duration_masks).sum())
        pitch_loss = pitch_loss.mul(variance_weights).masked_select(variance_masks).sum()
        energy_loss = (energy_loss.mul(variance_weights).masked_select(variance_masks).sum())

        return distance_loss, duration_loss, pitch_loss, energy_loss
