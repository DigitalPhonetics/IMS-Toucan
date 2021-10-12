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
            output_activation=None,
            use_batch_norm=True,
            use_concate=True,
            use_residual=False,
            reduction_factor=1,
            spk_embed_dim=None,
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
            use_dtw_loss=False,
            use_alignment_loss=True,
            input_layer_type="linear",
            init_type=None,
            initialize_from_pretrained_embedding_weights=False,
            initialize_encoder_from_pretrained_model=False,
            initialize_decoder_from_pretrained_model=False,
            initialize_multispeaker_projection=False,
            language_embedding_amount=None,  # pass None to not use language embeddings (training single-language models without meta-checkpoint) (default 30)
            speaker_embedding_projection_size=64):
        super().__init__()

        # store hyperparameters
        self.use_dtw_loss = use_dtw_loss
        self.use_alignment_loss = use_alignment_loss
        self.idim = idim
        self.odim = odim
        self.eos = idim - 1
        self.spk_embed_dim = spk_embed_dim
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

        self.language_embedding = None
        if language_embedding_amount is not None:
            self.language_embedding = torch.nn.Embedding(language_embedding_amount, embed_dim)

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

        if spk_embed_dim is not None:
            self.encoder_speakerembedding_projection = torch.nn.Linear(eunits + speaker_embedding_projection_size, eunits)
            # embedding projection derived from https://arxiv.org/pdf/1705.08947.pdf
            self.embedding_projection = torch.nn.Sequential(torch.nn.Linear(spk_embed_dim, speaker_embedding_projection_size),
                                                            torch.nn.Softsign())
        else:
            speaker_embedding_projection_size = None
        dec_idim = eunits

        loc_att = AttLoc(dec_idim, dunits, adim, aconv_chans, aconv_filts)

        forward_att = AttForwardTA(dec_idim, dunits, adim, aconv_chans, aconv_filts, odim)

        self.dec = Decoder(idim=dec_idim,
                           odim=odim,
                           loc_att=loc_att,
                           forward_att=forward_att,
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
                           reduction_factor=reduction_factor,
                           speaker_embedding_projection_size=speaker_embedding_projection_size)

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

        if init_type == "xavier_uniform":
            initialize(self, "xavier_uniform")  # doesn't go together well with forward attention
        if initialize_from_pretrained_embedding_weights:
            self.enc.embed.load_state_dict(torch.load("Preprocessing/embedding_pretrained_weights_combined_512dim.pt", map_location='cpu')["embedding_weights"])
        if initialize_encoder_from_pretrained_model:
            self.enc.load_state_dict(torch.load("Models/PretrainedModelTaco/enc.pt", map_location='cpu'))
        if initialize_decoder_from_pretrained_model:
            self.dec.load_state_dict(torch.load("Models/PretrainedModelTaco/dec.pt", map_location='cpu'))
        if initialize_multispeaker_projection:
            self.projection.load_state_dict(torch.load("Models/PretrainedModelTaco/projection.pt", map_location='cpu'))

    def forward(self,
                text,
                text_lengths,
                speech,
                speech_lengths,
                step,
                speaker_embeddings=None,
                prior=None,
                language_id=None,
                return_mels=False,
                return_loss_dict=False):
        """
        Calculate forward propagation.

        Args:
            step: current number of update steps taken as indicator when to start binarizing
            language_id: batch of lookup IDs for language embedding vectors
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

        losses = dict()

        # make labels for stop prediction
        labels = make_pad_mask(speech_lengths - 1).to(speech.device, speech.dtype)
        labels = F.pad(labels, [0, 1], "constant", 1.0)

        # calculate tacotron2 outputs
        after_outs, before_outs, logits, att_ws, att_ws_loc, att_ws_for = self._forward(text,
                                                                                        text_lengths,
                                                                                        speech,
                                                                                        speech_lengths,
                                                                                        speaker_embeddings,
                                                                                        prior=prior,
                                                                                        language_id=language_id)

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
            raise ValueError(f"unknown --loss-type {self.loss_type}")

        # calculate dtw loss
        if self.use_dtw_loss:
            dtw_loss = self.dtw_criterion(after_outs, speech).mean() / 2000.0  # division to balance orders of magnitude
            loss = loss + dtw_loss
            losses["dtw"] = dtw_loss.item()

        # calculate attention loss
        if self.use_guided_attn_loss:
            if self.reduction_factor > 1:
                olens_in = speech_lengths.new([olen // self.reduction_factor for olen in speech_lengths])
            else:
                olens_in = speech_lengths
            attn_loss_weight = min(1.0, 15.0 / max((step / 200.0), 1.0))
            attn_loss_loc = self.guided_att_loss(att_ws_loc, text_lengths, olens_in)
            attn_loss_for = self.guided_att_loss(att_ws_for, text_lengths, olens_in)
            losses["prior"] = attn_loss_loc.item() + attn_loss_for.item()
            loss = loss + ((attn_loss_loc + attn_loss_for) * attn_loss_weight)

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
                 speech_lengths,
                 speaker_embeddings,
                 prior=None,
                 language_id=None):
        if language_id is not None:
            language_embedding_vector = self.language_embedding(language_id)
            hs, hlens = self.enc(text_tensors, ilens, language_embedding=language_embedding_vector)
        else:
            hs, hlens = self.enc(text_tensors, ilens)
        if self.spk_embed_dim is not None:
            projected_speaker_embeddings = self.embedding_projection(speaker_embeddings)
            hs = self._integrate_with_spk_embed(hs, projected_speaker_embeddings)
        else:
            projected_speaker_embeddings = None
        return self.dec(hs, hlens, ys, projected_speaker_embeddings, prior=prior)

    def inference(self,
                  text_tensor,
                  speech_tensor=None,
                  speaker_embeddings=None,
                  threshold=0.5,
                  minlenratio=0.0,
                  maxlenratio=10.0,
                  use_att_constraint=False,
                  backward_window=1,
                  forward_window=3,
                  use_teacher_forcing=False,
                  language_id=None,
                  return_atts=False):
        """
        Generate the sequence of features given the sequences of characters.

        Args:
            language_id: lookup ID for language embedding vectors
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
            outs, _, _, att_ws = self._forward(text_tensors, ilens, speech_tensors, speech_lengths, speaker_embeddings, return_atts=return_atts)

            return outs[0], None, att_ws[0]

        # inference
        if language_id is not None:
            language_embedding_vector = self.language_embedding(language_id.view(-1))
            h = self.enc.inference(text_tensor, language_embedding=language_embedding_vector)
        else:
            h = self.enc.inference(text_tensor)

        if self.spk_embed_dim is not None:
            projected_speaker_embedding = self.embedding_projection(speaker_embedding)
            hs, speaker_embeddings = h.unsqueeze(0), projected_speaker_embedding.unsqueeze(0)
            h = self._integrate_with_spk_embed(hs, speaker_embeddings)[0]
        else:
            speaker_embeddings = None
        if return_atts:
            return self.dec.inference(h,
                                      threshold=threshold,
                                      minlenratio=minlenratio,
                                      maxlenratio=maxlenratio,
                                      use_att_constraint=use_att_constraint,
                                      backward_window=backward_window,
                                      forward_window=forward_window,
                                      speaker_embedding=speaker_embeddings,
                                      return_atts=return_atts)

        return self.dec.inference(h,
                                  threshold=threshold,
                                  minlenratio=minlenratio,
                                  maxlenratio=maxlenratio,
                                  use_att_constraint=use_att_constraint,
                                  backward_window=backward_window,
                                  forward_window=forward_window,
                                  speaker_embedding=speaker_embeddings,
                                  return_atts=return_atts)

    def _integrate_with_spk_embed(self, hs, speaker_embeddings):
        """
        Integrate speaker embedding with hidden states.

        Args:
            hs (Tensor): Batch of hidden state sequences (B, Tmax, adim).
            speaker_embeddings (Tensor): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Tensor: Batch of integrated hidden state sequences (B, Tmax, adim).

        """
        # concat hidden states with spk embeds and then apply projection
        speaker_embeddings_expanded = F.normalize(speaker_embeddings).unsqueeze(1).expand(-1, hs.size(1), -1)
        hs = self.encoder_speakerembedding_projection(torch.cat([hs, speaker_embeddings_expanded], dim=-1))
        return hs
