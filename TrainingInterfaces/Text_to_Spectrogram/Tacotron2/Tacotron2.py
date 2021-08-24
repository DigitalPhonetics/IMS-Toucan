# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Adapted by Florian Lux 2021

import torch
import torch.nn.functional as F

from Layers.Attention import GuidedAttentionLoss
from Layers.RNNAttention import AttForward
from Layers.RNNAttention import AttForwardTA
from Layers.RNNAttention import AttLoc
from Layers.TacotronDecoder import Decoder
from Layers.TacotronEncoder import Encoder
from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.Tacotron2Loss import Tacotron2Loss
from Utility.SoftDTW.sdtw_cuda_loss import SoftDTW
from Utility.utils import initialize
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
            idim=25,  # 24 articulatory features + 1 feature to indicate a pause
            odim=80,
            embed_dim=512,
            elayers=1,
            eunits=512,
            econv_layers=3,
            econv_chans=512,
            econv_filts=5,
            atype="forward_ta",
            adim=512,
            aconv_chans=32,
            aconv_filts=15,
            cumulate_att_w=True,
            dlayers=2,
            dunits=1024,
            prenet_layers=2,
            prenet_units=256,
            postnet_layers=5,
            postnet_chans=512,
            postnet_filts=5,
            output_activation=None,
            use_batch_norm=True,
            use_concate=True,
            use_residual=False,
            reduction_factor=1,
            spk_embed_dim=None,
            spk_embed_integration_type="concat",
            # training related
            dropout_rate=0.5,
            zoneout_rate=0.1,
            use_masking=False,
            use_weighted_masking=True,
            bce_pos_weight=10.0,
            loss_type="L1+L2",
            use_guided_attn_loss=True,
            guided_attn_loss_sigma=0.2,
            guided_attn_loss_lambda=10.0,
            guided_attn_loss_lambda_later=1.0,
            guided_attn_loss_sigma_later=0.4,
            use_dtw_loss=False,
            input_layer_type="linear",
            start_with_prenet=False,
            switch_on_prenet_step=20000):
        super().__init__()

        # store hyperparameters
        self.use_dtw_loss = use_dtw_loss
        self.switch_on_prenet_step = switch_on_prenet_step
        self.prenet_on = start_with_prenet
        self.idim = idim
        self.odim = odim
        self.eos = idim - 1
        self.spk_embed_dim = spk_embed_dim
        self.cumulate_att_w = cumulate_att_w
        self.reduction_factor = reduction_factor
        self.use_guided_attn_loss = use_guided_attn_loss
        self.loss_type = loss_type
        if self.spk_embed_dim is not None:
            self.spk_embed_integration_type = spk_embed_integration_type

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
                           dropout_rate=dropout_rate, )

        if spk_embed_dim is None:
            dec_idim = eunits
        elif spk_embed_integration_type == "concat":
            dec_idim = eunits + spk_embed_dim
        elif spk_embed_integration_type == "add":
            dec_idim = eunits
            self.projection = torch.nn.Linear(self.spk_embed_dim, eunits)
        else:
            raise ValueError(f"{spk_embed_integration_type} is not supported.")

        if atype == "location":
            att = AttLoc(dec_idim, dunits, adim, aconv_chans, aconv_filts)
        elif atype == "forward":
            att = AttForward(dec_idim, dunits, adim, aconv_chans, aconv_filts)
            if self.cumulate_att_w:
                self.cumulate_att_w = False
        elif atype == "forward_ta":
            att = AttForwardTA(dec_idim, dunits, adim, aconv_chans, aconv_filts, odim)
            if self.cumulate_att_w:
                self.cumulate_att_w = False
        else:
            raise NotImplementedError("Support only location or forward")
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
                           cumulate_att_w=self.cumulate_att_w,
                           use_batch_norm=use_batch_norm,
                           use_concate=use_concate,
                           dropout_rate=dropout_rate,
                           zoneout_rate=zoneout_rate,
                           reduction_factor=reduction_factor,
                           start_with_prenet=start_with_prenet)
        self.taco2_loss = Tacotron2Loss(use_masking=use_masking,
                                        use_weighted_masking=use_weighted_masking,
                                        bce_pos_weight=bce_pos_weight, )
        if self.use_guided_attn_loss:
            self.attn_loss = GuidedAttentionLoss(sigma=guided_attn_loss_sigma,
                                                 alpha=guided_attn_loss_lambda, )
            self.attn_loss_later = GuidedAttentionLoss(sigma=guided_attn_loss_sigma_later,
                                                       alpha=guided_attn_loss_lambda_later, )
        if self.use_dtw_loss:
            self.dtw_criterion = SoftDTW(use_cuda=True, gamma=0.1)

        initialize(self, "xavier_uniform")

    def forward(self,
                text: torch.Tensor,
                text_lengths: torch.Tensor,
                speech: torch.Tensor,
                speech_lengths: torch.Tensor,
                speaker_embeddings: torch.Tensor = None,
                step=None):
        """
        Calculate forward propagation.

        Args:
            step: Indicator for when to relax the attention constraint
            text (LongTensor): Batch of padded character ids (B, Tmax).
            text_lengths (LongTensor): Batch of lengths of each input batch (B,).
            speech (Tensor): Batch of padded target features (B, Lmax, odim).
            speech_lengths (LongTensor): Batch of the lengths of each target (B,).
            speaker_embeddings (Tensor, optional): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Tensor: Loss scalar value.
            Dict: Statistics to be monitored.
            Tensor: Weight value.
        """

        # For the articulatory frontend, EOS is already added as last of the sequence in preprocessing
        # eos is [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # make labels for stop prediction
        labels = make_pad_mask(speech_lengths - 1).to(speech.device, speech.dtype)
        labels = F.pad(labels, [0, 1], "constant", 1.0)

        # calculate tacotron2 outputs
        after_outs, before_outs, logits, att_ws = self._forward(text, text_lengths, speech, speech_lengths, speaker_embeddings)

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
        elif self.loss_type == "L1":
            loss = l1_loss + bce_loss
        elif self.loss_type == "L2":
            loss = mse_loss + bce_loss
        else:
            raise ValueError(f"unknown --loss-type {self.loss_type}")

        if self.use_dtw_loss:
            print("Regular Loss: {}".format(loss))
            dtw_loss = self.dtw_criterion(after_outs, speech).mean() / 2000.0  # division to balance orders of magnitude
            # print("\n\n")
            # import matplotlib.pyplot as plt
            # import librosa.display as lbd
            # fig, ax = plt.subplots(nrows=2, ncols=1)
            # lbd.specshow(after_outs[0].transpose(0,1).detach().cpu().numpy(), ax=ax[0], sr=16000, cmap='GnBu', y_axis='mel', x_axis='time', hop_length=256)
            # lbd.specshow(speech[0].transpose(0,1).cpu().numpy(), ax=ax[1], sr=16000, cmap='GnBu', y_axis='mel', x_axis='time', hop_length=256)
            # plt.show()
            print("DTW Loss: {}".format(dtw_loss))
            loss += dtw_loss

        # calculate attention loss
        if self.use_guided_attn_loss:
            # NOTE(kan-bayashi): length of output for auto-regressive
            # input will be changed when r > 1
            if self.reduction_factor > 1:
                speech_lengths_in = speech_lengths.new([olen // self.reduction_factor for olen in speech_lengths])
            else:
                speech_lengths_in = speech_lengths

            if step is not None:
                if step < 5000:
                    attn_loss = self.attn_loss(att_ws, text_lengths, speech_lengths_in)
                else:
                    attn_loss = self.attn_loss_later(att_ws, text_lengths, speech_lengths_in)
                if step > self.switch_on_prenet_step and not self.prenet_on:
                    self.prenet_on = True
                    self.dec.add_prenet()
            else:
                attn_loss = self.attn_loss(att_ws, text_lengths, speech_lengths_in)
            loss = loss + attn_loss

        return loss

    def _forward(self,
                 text_tensors: torch.Tensor,
                 ilens: torch.Tensor,
                 ys: torch.Tensor,
                 speech_lengths: torch.Tensor,
                 speaker_embeddings: torch.Tensor, ):
        hs, hlens = self.enc(text_tensors, ilens)
        if self.spk_embed_dim is not None:
            hs = self._integrate_with_spk_embed(hs, speaker_embeddings)
        return self.dec(hs, hlens, ys)

    def inference(self,
                  text_tensor: torch.Tensor,
                  speech_tensor: torch.Tensor = None,
                  speaker_embeddings: torch.Tensor = None,
                  threshold: float = 0.5,
                  minlenratio: float = 0.0,
                  maxlenratio: float = 10.0,
                  use_att_constraint: bool = False,
                  backward_window: int = 1,
                  forward_window: int = 3,
                  use_teacher_forcing: bool = False, ):
        """
        Generate the sequence of features given the sequences of characters.

        Args:
            text_tensor (LongTensor): Input sequence of characters (T,).
            speech_tensor (Tensor, optional): Feature sequence to extract style (N, idim).
            speaker_embeddings (Tensor, optional): Speaker embedding vector (spk_embed_dim,).
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
        speaker_embedding = speaker_embeddings

        # inference with teacher forcing
        if use_teacher_forcing:
            assert speech_tensor is not None, "speech must be provided with teacher forcing."

            text_tensors, speech_tensors = text_tensor.unsqueeze(0), speech_tensor.unsqueeze(0)
            speaker_embeddings = None if speaker_embedding is None else speaker_embedding.unsqueeze(0)
            ilens = text_tensor.new_tensor([text_tensors.size(1)]).long()
            speech_lengths = speech_tensor.new_tensor([speech_tensors.size(1)]).long()
            outs, _, _, att_ws = self._forward(text_tensors, ilens, speech_tensors, speech_lengths, speaker_embeddings)

            return outs[0], None, att_ws[0]

        # inference
        h = self.enc.inference(text_tensor)
        if self.spk_embed_dim is not None:
            hs, speaker_embeddings = h.unsqueeze(0), speaker_embedding.unsqueeze(0)
            h = self._integrate_with_spk_embed(hs, speaker_embeddings)[0]
        outs, probs, att_ws = self.dec.inference(h,
                                                 threshold=threshold,
                                                 minlenratio=minlenratio,
                                                 maxlenratio=maxlenratio,
                                                 use_att_constraint=use_att_constraint,
                                                 backward_window=backward_window,
                                                 forward_window=forward_window, )

        return outs, probs, att_ws

    def _integrate_with_spk_embed(self, hs: torch.Tensor,
                                  speaker_embeddings: torch.Tensor):
        """
        Integrate speaker embedding with hidden states.

        Args:
            hs (Tensor): Batch of hidden state sequences (B, Tmax, eunits).
            speaker_embeddings (Tensor): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Tensor: Batch of integrated hidden state sequences (B, Tmax, eunits) if
                integration_type is "add" else (B, Tmax, eunits + spk_embed_dim).
        """
        if self.spk_embed_integration_type == "add":
            # apply projection and then add to hidden states
            speaker_embeddings = self.projection(F.normalize(speaker_embeddings))
            hs = hs + speaker_embeddings.unsqueeze(1)
        elif self.spk_embed_integration_type == "concat":
            # concat hidden states with spk embeds
            speaker_embeddings = F.normalize(speaker_embeddings).unsqueeze(1).expand(-1, hs.size(1), -1)
            hs = torch.cat([hs, speaker_embeddings], dim=-1)
        else:
            raise NotImplementedError("support only add or concat.")

        return hs
