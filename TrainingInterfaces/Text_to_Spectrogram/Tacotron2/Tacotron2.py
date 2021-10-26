# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Adapted by Florian Lux 2021

import torch
import torch.nn.functional as F

from Layers.Attention import GuidedAttentionLoss
from Layers.RNNAttention import AttForwardTA
from Layers.RNNAttention import AttLoc
from Layers.TacotronDecoder import Decoder
from Layers.TacotronEncoder import Encoder
from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.AlignmentLoss import AlignmentLoss
from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.Tacotron2Loss import Tacotron2Loss
from Utility.SoftDTW.sdtw_cuda_loss import SoftDTW
from Utility.utils import make_pad_mask


class Tacotron2(torch.nn.Module):
    """
    Tacotron2 module.

    This is a module of Spectrogram prediction network in Tacotron2

    .. _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`:
       https://arxiv.org/abs/1712.05884
   """

    def __init__(
            self,
            # network structure related
            idim=66,  # 24 articulatory features from PanPhon, 42 from Papercup (one-hot)
            odim=80,
            embed_dim=512,
            elayers=1,
            eunits=512,
            econv_layers=3,
            econv_chans=512,
            econv_filts=5,
            adim=512,
            aconv_chans=32,
            aconv_filts=15,
            cumulate_att_w=True,
            dlayers=2,
            dunits=1024,
            prenet_layers=2,
            prenet_units=256,  # default in the paper is 256, but can cause over-reliance on teacher forcing, so 64 sometimes recommended
            postnet_layers=5,
            postnet_chans=512,
            postnet_filts=5,
            attention_type="forward",
            output_activation=None,
            use_batch_norm=True,
            use_concate=True,
            use_residual=False,
            reduction_factor=1,
            # training related
            dropout_rate=0.5,
            zoneout_rate=0.1,
            use_masking=False,
            use_weighted_masking=True,
            bce_pos_weight=20.0,
            loss_type="L1+L2",
            use_guided_attn_loss=True,
            guided_attn_loss_lambda=1.0,  # weight of the attention loss
            guided_attn_loss_sigma=0.4,  # deviation from the main diagonal that is allowed
            use_dtw_loss=False,  # really cool concept, but requires tons and tons of GPU-memory
            use_alignment_loss=False,
            input_layer_type="linear"):
        super().__init__()

        # store hyperparameters
        self.use_dtw_loss = use_dtw_loss
        self.use_alignment_loss = use_alignment_loss
        self.idim = idim
        self.odim = odim
        self.eos = idim - 1
        self.cumulate_att_w = cumulate_att_w
        self.reduction_factor = reduction_factor
        self.use_guided_attn_loss = use_guided_attn_loss
        self.loss_type = loss_type

        # define activation function for the final output
        if output_activation is None:
            self.output_activation_fn = None
        elif hasattr(F, output_activation):
            self.output_activation_fn = getattr(F, output_activation)
        else:
            raise ValueError(f"there is no such activation function. " f"({output_activation})")

        # set padding idx
        self.padding_idx = torch.zeros(idim)

        # define network modules
        self.enc = Encoder(idim=idim,
                           input_layer=input_layer_type,
                           embed_dim=embed_dim,
                           elayers=elayers,
                           eunits=eunits,
                           econv_layers=econv_layers,
                           econv_chans=econv_chans,
                           econv_filts=econv_filts,
                           use_batch_norm=use_batch_norm,
                           use_residual=use_residual,
                           dropout_rate=dropout_rate)

        if elayers == 0:
            dec_idim = embed_dim
        else:
            dec_idim = eunits

        if attention_type == "location":
            att = AttLoc(dec_idim, dunits, adim, aconv_chans, aconv_filts)
        elif attention_type == "forward":
            att = AttForwardTA(dec_idim, dunits, adim, aconv_chans, aconv_filts, odim)
        else:
            raise ValueError(f"unknown attention_type: {attention_type}")

        self.dec = Decoder(idim=dec_idim,
                           odim=odim,
                           att=att,
                           dlayers=dlayers,
                           dunits=dunits,
                           prenet_layers=prenet_layers,
                           prenet_units=prenet_units,
                           postnet_layers=postnet_layers,
                           postnet_chans=postnet_chans,
                           postnet_filts=postnet_filts,
                           output_activation_fn=self.output_activation_fn,
                           use_batch_norm=use_batch_norm,
                           use_concate=use_concate,
                           dropout_rate=dropout_rate,
                           zoneout_rate=zoneout_rate,
                           reduction_factor=reduction_factor)

        self.taco2_loss = Tacotron2Loss(use_masking=use_masking,
                                        use_weighted_masking=use_weighted_masking,
                                        bce_pos_weight=bce_pos_weight, )
        if self.use_guided_attn_loss:
            self.guided_att_loss = GuidedAttentionLoss(sigma=guided_attn_loss_sigma,
                                                       alpha=guided_attn_loss_lambda)
        if self.use_dtw_loss:
            self.dtw_criterion = SoftDTW(use_cuda=True, gamma=0.1)

        if self.use_alignment_loss:
            self.alignment_loss = AlignmentLoss()

    def forward(self,
                text,
                text_lengths,
                speech,
                speech_lengths,
                step,
                return_mels=False,
                return_loss_dict=False):
        """
        Calculate forward propagation.

        Args:
            step: current number of update steps taken as indicator when to start binarizing
            text (LongTensor): Batch of padded character ids (B, Tmax).
            text_lengths (LongTensor): Batch of lengths of each input batch (B,).
            speech (Tensor): Batch of padded target features (B, Lmax, odim).
            speech_lengths (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: Loss scalar value.
            Dict: Statistics to be monitored.
            Tensor: Weight value.
        """

        # For the articulatory frontend, EOS is already added as last of the sequence in preprocessing
        losses = dict()

        # make labels for stop prediction
        labels = make_pad_mask(speech_lengths - 1).to(speech.device, speech.dtype)
        labels = F.pad(labels, [0, 1], "constant", 1.0)

        # calculate tacotron2 outputs
        after_outs, before_outs, logits, att_ws = self._forward(text,
                                                                text_lengths,
                                                                speech,
                                                                speech_lengths)

        # modify mod part of groundtruth
        if self.reduction_factor > 1:
            assert speech_lengths.ge(self.reduction_factor).all(), "Output length must be greater than or equal to reduction factor."
            speech_lengths = speech_lengths.new([olen - olen % self.reduction_factor for olen in speech_lengths])
            max_out = max(speech_lengths)
            speech = speech[:, :max_out]
            labels = labels[:, :max_out]
            labels = torch.scatter(labels, 1, (speech_lengths - 1).unsqueeze(1), 1.0)  # see #3388

        # calculate taco2 loss
        l1_loss, mse_loss, bce_loss = self.taco2_loss(after_outs, before_outs, logits, speech, labels, speech_lengths)
        if self.loss_type == "L1+L2":
            loss = l1_loss + mse_loss + bce_loss
            losses["l1"] = l1_loss.item()
            losses["mse"] = mse_loss.item()
            losses["bce"] = bce_loss.item()
        else:
            raise ValueError(f"unknown loss-type {self.loss_type}")

        # calculate dtw loss
        if self.use_dtw_loss:
            if len(speech[0]) < 1024:
                # max block size supported by cuda. Have to skip this batch if sequence is too long
                dtw_loss = self.dtw_criterion(after_outs, speech).mean() / 2000.0  # division to balance orders of magnitude
                loss = loss + dtw_loss
                losses["dtw"] = dtw_loss.item()

        # calculate attention loss
        if self.use_guided_attn_loss:
            if self.reduction_factor > 1:
                olens_in = speech_lengths.new([olen // self.reduction_factor for olen in speech_lengths])
            else:
                olens_in = speech_lengths
            attn_loss_weight = max(1.0, 10.0 / max((step / 200.0), 1.0))
            attn_loss = self.guided_att_loss(att_ws, text_lengths, olens_in) * attn_loss_weight
            losses["diag"] = attn_loss.item()
            loss = loss + attn_loss

        # calculate alignment loss
        if self.use_alignment_loss:
            if self.reduction_factor > 1:
                olens_in = speech_lengths.new([olen // self.reduction_factor for olen in speech_lengths])
            else:
                olens_in = speech_lengths
            align_loss = self.alignment_loss(att_ws, text_lengths, olens_in, step)
            if align_loss != 0.0:
                losses["align"] = align_loss.item()
                loss = loss + align_loss

        if return_mels:
            if return_loss_dict:
                return loss, after_outs, losses
            return loss, after_outs
        if return_loss_dict:
            return loss, losses
        return loss

    def _forward(self,
                 text_tensors,
                 ilens,
                 ys,
                 speech_lengths):
        hs, hlens = self.enc(text_tensors, ilens)
        return self.dec(hs, hlens, ys)

    def inference(self,
                  text_tensor,
                  speech_tensor=None,
                  threshold=0.5,
                  minlenratio=0.0,
                  maxlenratio=10.0,
                  use_att_constraint=False,
                  backward_window=1,
                  forward_window=3,
                  use_teacher_forcing=False):
        """
        Generate the sequence of features given the sequences of characters.

        Args:
            text_tensor (LongTensor): Input sequence of characters (T,).
            speech_tensor (Tensor, optional): Feature sequence to extract style (N, idim).
            threshold (float, optional): Threshold in inference.
            minlenratio (float, optional): Minimum length ratio in inference.
            maxlenratio (float, optional): Maximum length ratio in inference.
            use_att_constraint (bool, optional): Whether to apply attention constraint.
            backward_window (int, optional): Backward window in attention constraint.
            forward_window (int, optional): Forward window in attention constraint.
            use_teacher_forcing (bool, optional): Whether to use teacher forcing.

        Returns:
            Tensor: Output sequence of features (L, odim).
            Tensor: Output sequence of stop probabilities (L,).
            Tensor: Attention weights (L, T).
        """

        # inference with teacher forcing
        if use_teacher_forcing:
            assert speech_tensor is not None, "speech must be provided with teacher forcing."

            text_tensors, speech_tensors = text_tensor.unsqueeze(0), speech_tensor.unsqueeze(0)
            ilens = text_tensor.new_tensor([text_tensors.size(1)], device=text_tensor.device).long()
            speech_lengths = speech_tensor.new_tensor([speech_tensors.size(1)], device=text_tensor.device).long()
            outs, _, _, att_ws = self._forward(text_tensors, ilens, speech_tensors, speech_lengths)

            return outs[0], None, att_ws[0]

        # inference
        h = self.enc.inference(text_tensor)

        return self.dec.inference(h,
                                  threshold=threshold,
                                  minlenratio=minlenratio,
                                  maxlenratio=maxlenratio,
                                  use_att_constraint=use_att_constraint,
                                  backward_window=backward_window,
                                  forward_window=forward_window)
