# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Adapted by Florian Lux 2021

from abc import ABC
from typing import Dict

import torch
import torch.nn.functional as F

from Layers.Attention import GuidedMultiHeadAttentionLoss
from Layers.Attention import MultiHeadedAttention
from Layers.PositionalEncoding import PositionalEncoding
from Layers.PositionalEncoding import ScaledPositionalEncoding
from Layers.PostNet import PostNet
from Layers.TransformerTTSDecoder import Decoder
from Layers.TransformerTTSDecoderPrenet import DecoderPrenet
from Layers.TransformerTTSEncoder import Encoder
from Layers.TransformerTTSEncoderPrenet import EncoderPrenet
from TrainingInterfaces.Text_to_Spectrogram.Tacotron2 import TransformerLoss
from Utility.SoftDTW.sdtw_cuda_loss import SoftDTW
from Utility.utils import initialize
from Utility.utils import make_non_pad_mask
from Utility.utils import make_pad_mask
from Utility.utils import subsequent_mask


class Transformer(torch.nn.Module, ABC):
    """
    TTS-Transformer module.

    This is a module of text-to-speech Transformer described in `Neural Speech Synthesis
    with Transformer Network`_, which convert the sequence of tokens into the sequence
    of Mel-filterbanks.

    .. _`Neural Speech Synthesis with Transformer Network`:
        https://arxiv.org/pdf/1809.08895.pdf
    """

    def __init__(self,
                 # network structure related
                 idim,
                 odim,
                 embed_dim=0,
                 eprenet_conv_layers=0,
                 eprenet_conv_chans=0,
                 eprenet_conv_filts=0,
                 dprenet_layers=2,
                 dprenet_units=256,
                 elayers=6,
                 eunits=1024,
                 adim=512,
                 aheads=4,
                 dlayers=6,
                 dunits=1024,
                 postnet_layers=5,
                 postnet_chans=256,
                 postnet_filts=5,
                 positionwise_layer_type="conv1d",
                 positionwise_conv_kernel_size=1,
                 use_scaled_pos_enc=True,
                 use_batch_norm=True,
                 encoder_normalize_before=True,
                 decoder_normalize_before=True,
                 encoder_concat_after=True,  # True is better according to https://github.com/soobinseo/Transformer-TTS
                 decoder_concat_after=True,  # True is better according to https://github.com/soobinseo/Transformer-TTS
                 reduction_factor=1,
                 spk_embed_dim=None,
                 spk_embed_integration_type="concat",
                 # training related
                 transformer_enc_dropout_rate=0.1,
                 transformer_enc_positional_dropout_rate=0.1,
                 transformer_enc_attn_dropout_rate=0.1,
                 transformer_dec_dropout_rate=0.1,
                 transformer_dec_positional_dropout_rate=0.1,
                 transformer_dec_attn_dropout_rate=0.1,
                 transformer_enc_dec_attn_dropout_rate=0.1,
                 eprenet_dropout_rate=0.0,
                 dprenet_dropout_rate=0.5,
                 postnet_dropout_rate=0.5,
                 init_type="xavier_uniform",  # since we have little to no asymmetric activations, this seems to work better than kaiming
                 init_enc_alpha=1.0,
                 init_dec_alpha=1.0,
                 use_masking=False,  # either this or weighted masking, not both
                 use_weighted_masking=True,  # if there are severely different sized samples in one batch
                 bce_pos_weight=7.0,  # scaling the loss of the stop token prediction
                 loss_type="L1",
                 use_guided_attn_loss=True,
                 num_heads_applied_guided_attn=2,
                 num_layers_applied_guided_attn=2,
                 modules_applied_guided_attn=("encoder-decoder",),
                 guided_attn_loss_sigma=0.4,  # standard deviation from diagonal that is allowed
                 guided_attn_loss_lambda=25.0,  # forcing the attention to be diagonal
                 # additional features
                 legacy_model=False,
                 use_dtw_loss=False):
        super().__init__()

        # store hyperparameters
        self.use_dtw_loss = use_dtw_loss
        self.idim = idim
        self.odim = odim
        self.eos = 1
        self.aheads = aheads
        self.adim = adim
        self.spk_embed_dim = spk_embed_dim
        self.reduction_factor = reduction_factor
        self.use_guided_attn_loss = use_guided_attn_loss
        self.use_scaled_pos_enc = use_scaled_pos_enc
        self.loss_type = loss_type
        self.use_guided_attn_loss = use_guided_attn_loss
        if self.use_guided_attn_loss:
            if num_layers_applied_guided_attn == -1:
                self.num_layers_applied_guided_attn = elayers
            else:
                self.num_layers_applied_guided_attn = num_layers_applied_guided_attn
            if num_heads_applied_guided_attn == -1:
                self.num_heads_applied_guided_attn = aheads
            else:
                self.num_heads_applied_guided_attn = num_heads_applied_guided_attn
            self.modules_applied_guided_attn = modules_applied_guided_attn
        if self.spk_embed_dim is not None:
            self.spk_embed_integration_type = spk_embed_integration_type

        # use idx 0 as padding idx
        self.padding_idx = 0

        # get positional encoding class
        pos_enc_class = (ScaledPositionalEncoding if self.use_scaled_pos_enc else PositionalEncoding)

        # define transformer encoder
        if eprenet_conv_layers != 0:
            # encoder prenet
            encoder_input_layer = torch.nn.Sequential(
                EncoderPrenet(idim=idim, embed_dim=embed_dim, elayers=0, econv_layers=eprenet_conv_layers, econv_chans=eprenet_conv_chans,
                              econv_filts=eprenet_conv_filts, use_batch_norm=use_batch_norm, dropout_rate=eprenet_dropout_rate, padding_idx=self.padding_idx),
                torch.nn.Linear(eprenet_conv_chans, adim))
        else:
            encoder_input_layer = torch.nn.Embedding(num_embeddings=idim, embedding_dim=adim, padding_idx=self.padding_idx)
        self.encoder = Encoder(idim=idim, attention_dim=adim, attention_heads=aheads, linear_units=eunits, num_blocks=elayers, input_layer=encoder_input_layer,
                               dropout_rate=transformer_enc_dropout_rate, positional_dropout_rate=transformer_enc_positional_dropout_rate,
                               attention_dropout_rate=transformer_enc_attn_dropout_rate, pos_enc_class=pos_enc_class, normalize_before=encoder_normalize_before,
                               concat_after=encoder_concat_after, positionwise_layer_type=positionwise_layer_type,
                               positionwise_conv_kernel_size=positionwise_conv_kernel_size)

        # define projection layer
        if self.spk_embed_dim is not None:
            if self.spk_embed_integration_type == "add":
                self.projection = torch.nn.Linear(self.spk_embed_dim, adim)
            else:
                self.projection = torch.nn.Linear(adim + self.spk_embed_dim, adim)

        # define transformer decoder
        if dprenet_layers != 0:
            # decoder prenet
            decoder_input_layer = torch.nn.Sequential(
                DecoderPrenet(idim=odim, n_layers=dprenet_layers, n_units=dprenet_units, dropout_rate=dprenet_dropout_rate),
                torch.nn.Linear(dprenet_units, adim))
        else:
            decoder_input_layer = "linear"
        self.decoder = Decoder(odim=odim,  # odim is needed when no prenet is used
                               attention_dim=adim, attention_heads=aheads, linear_units=dunits, num_blocks=dlayers, dropout_rate=transformer_dec_dropout_rate,
                               positional_dropout_rate=transformer_dec_positional_dropout_rate, self_attention_dropout_rate=transformer_dec_attn_dropout_rate,
                               src_attention_dropout_rate=transformer_enc_dec_attn_dropout_rate, input_layer=decoder_input_layer, use_output_layer=False,
                               pos_enc_class=pos_enc_class, normalize_before=decoder_normalize_before, concat_after=decoder_concat_after)

        # define final projection
        self.feat_out = torch.nn.Linear(adim, odim * reduction_factor)
        self.prob_out = torch.nn.Linear(adim, reduction_factor)

        # define postnet
        self.postnet = PostNet(idim=idim, odim=odim, n_layers=postnet_layers, n_chans=postnet_chans, n_filts=postnet_filts, use_batch_norm=use_batch_norm,
                               dropout_rate=postnet_dropout_rate, legacy_model=legacy_model)

        # define loss function
        self.criterion = TransformerLoss(use_masking=use_masking, use_weighted_masking=use_weighted_masking, bce_pos_weight=bce_pos_weight)
        if self.use_guided_attn_loss:
            self.attn_criterion = GuidedMultiHeadAttentionLoss(sigma=guided_attn_loss_sigma, alpha=guided_attn_loss_lambda)
        self.dtw_criterion = SoftDTW(use_cuda=True, gamma=0.1)

        # initialize parameters
        self._reset_parameters(init_type=init_type, init_enc_alpha=init_enc_alpha, init_dec_alpha=init_dec_alpha)

    def _reset_parameters(self, init_type, init_enc_alpha=1.0, init_dec_alpha=1.0):
        # initialize parameters
        if init_type != "pytorch":
            initialize(self, init_type)

        # initialize alpha in scaled positional encoding
        if self.use_scaled_pos_enc:
            self.encoder.embed[-1].alpha.data = torch.tensor(init_enc_alpha)
            self.decoder.embed[-1].alpha.data = torch.tensor(init_dec_alpha)

    def forward(self, text, text_lengths, speech, speech_lengths, speaker_embeddings=None):
        """
        Calculate forward propagation.

        Args:
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
        text = text[:, : text_lengths.max()]  # for data-parallel if I ever end up implementing that
        speech = speech[:, : speech_lengths.max()]  # for data-parallel if I ever end up implementing that

        # Add eos at the last of sequence
        xs = F.pad(text, [0, 1], "constant", self.padding_idx)
        for i, l in enumerate(text_lengths):
            xs[i, l] = self.eos

        # make labels for stop prediction
        labels = make_pad_mask(speech_lengths - 1).to(speech.device, speech.dtype)
        labels = F.pad(labels, [0, 1], "constant", 1.0)

        # calculate transformer outputs
        after_outs, before_outs, logits = self._forward(text, text_lengths, speech, speech_lengths, speaker_embeddings)

        # modify mod part of groundtruth
        olens_in = speech_lengths
        if self.reduction_factor > 1:
            olens_in = speech_lengths.new([olen // self.reduction_factor for olen in speech_lengths])
            speech_lengths = speech_lengths.new([olen - olen % self.reduction_factor for olen in speech_lengths])
            max_olen = max(speech_lengths)
            speech = speech[:, :max_olen]
            labels = labels[:, :max_olen]
            labels[:, -1] = 1.0  # make sure at least one frame has 1

        # calculate loss values
        l1_loss, l2_loss, bce_loss = self.criterion(after_outs, before_outs, logits, speech, labels, speech_lengths)
        if self.loss_type == "L1":
            loss = l1_loss + bce_loss
        elif self.loss_type == "L2":
            loss = l2_loss + bce_loss
        elif self.loss_type == "L1+L2":
            loss = l1_loss + l2_loss + bce_loss
        else:
            raise ValueError("unknown --loss-type " + self.loss_type)

        if self.use_dtw_loss:
            # print("Regular Loss: {}".format(loss))
            dtw_loss = self.dtw_criterion(after_outs, speech).mean() / 6000.0  # division to balance orders of magnitude
            # print("DTW Loss: {}".format(dtw_loss))
            loss += dtw_loss

        # calculate guided attention loss
        if self.use_guided_attn_loss:
            # calculate for encoder
            if "encoder" in self.modules_applied_guided_attn:
                att_ws = []
                for idx, layer_idx in enumerate(reversed(range(len(self.encoder.encoders)))):
                    att_ws += [self.encoder.encoders[layer_idx].self_attn.attn[:, : self.num_heads_applied_guided_attn]]
                    if idx + 1 == self.num_layers_applied_guided_attn:
                        break
                att_ws = torch.cat(att_ws, dim=1)  # (B, H*L, T_in, T_in)
                enc_attn_loss = self.attn_criterion(att_ws, text_lengths, text_lengths)
                loss = loss + enc_attn_loss
            # calculate for decoder
            if "decoder" in self.modules_applied_guided_attn:
                att_ws = []
                for idx, layer_idx in enumerate(reversed(range(len(self.decoder.decoders)))):
                    att_ws += [self.decoder.decoders[layer_idx].self_attn.attn[:, : self.num_heads_applied_guided_attn]]
                    if idx + 1 == self.num_layers_applied_guided_attn:
                        break
                att_ws = torch.cat(att_ws, dim=1)  # (B, H*L, T_out, T_out)
                dec_attn_loss = self.attn_criterion(att_ws, olens_in, olens_in)
                loss = loss + dec_attn_loss
            # calculate for encoder-decoder
            if "encoder-decoder" in self.modules_applied_guided_attn:
                att_ws = []
                for idx, layer_idx in enumerate(reversed(range(len(self.decoder.decoders)))):
                    att_ws += [self.decoder.decoders[layer_idx].src_attn.attn[:, : self.num_heads_applied_guided_attn]]
                    if idx + 1 == self.num_layers_applied_guided_attn:
                        break
                att_ws = torch.cat(att_ws, dim=1)  # (B, H*L, T_out, T_in)
                enc_dec_attn_loss = self.attn_criterion(att_ws, text_lengths, olens_in)
                loss = loss + enc_dec_attn_loss

        return loss

    def _forward(self, xs, ilens, ys, olens, speaker_embeddings):
        # forward encoder
        x_masks = self._source_mask(ilens)
        hs, h_masks = self.encoder(xs, x_masks)

        # integrate speaker embedding
        if self.spk_embed_dim is not None:
            hs = self._integrate_with_spk_embed(hs, speaker_embeddings)

        # thin out frames for reduction factor (B, Lmax, odim) ->  (B, Lmax//r, odim)
        if self.reduction_factor > 1:
            ys_in = ys[:, self.reduction_factor - 1:: self.reduction_factor]
            olens_in = olens.new([olen // self.reduction_factor for olen in olens])
        else:
            ys_in, olens_in = ys, olens

        # add first zero frame and remove last frame for auto-regressive
        ys_in = self._add_first_frame_and_remove_last_frame(ys_in)

        # forward decoder
        y_masks = self._target_mask(olens_in)
        zs, _ = self.decoder(ys_in, y_masks, hs, h_masks)
        # (B, Lmax//r, odim * r) -> (B, Lmax//r * r, odim)
        before_outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)
        # (B, Lmax//r, r) -> (B, Lmax//r * r)
        logits = self.prob_out(zs).view(zs.size(0), -1)

        # postnet -> (B, Lmax//r * r, odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(before_outs.transpose(1, 2)).transpose(1, 2)

        return after_outs, before_outs, logits

    def inference(self, text, speech=None, speaker_embeddings=None, threshold=0.5, minlenratio=0.0,
                  maxlenratio=10.0, use_teacher_forcing=False):
        """
        Generate the sequence of features given the sequences of characters.

        Args:
            text (LongTensor): Input sequence of characters (T,).
            speech (Tensor, optional): Feature sequence to extract style (N, idim).
            speaker_embeddings (Tensor, optional): Speaker embedding vector (spk_embed_dim,).
            threshold (float, optional): Threshold in inference.
            minlenratio (float, optional): Minimum length ratio in inference.
            maxlenratio (float, optional): Maximum length ratio in inference.
            use_teacher_forcing (bool, optional): Whether to use teacher forcing.

        Returns:
            Tensor: Output sequence of features (L, odim).
            Tensor: Output sequence of stop probabilities (L,).
            Tensor: Encoder-decoder (source) attention weights (#layers, #heads, L, T).
        """
        x = text
        y = speech
        speaker_embedding = speaker_embeddings
        self.eval()

        # add eos at the last of sequence
        x = F.pad(x, [0, 1], "constant", self.eos)

        # inference with teacher forcing
        if use_teacher_forcing:
            assert speech is not None, "speech must be provided with teacher forcing."

            # get teacher forcing outputs
            xs, ys = x.unsqueeze(0), y.unsqueeze(0)
            speaker_embeddings = None if speaker_embedding is None else speaker_embedding.unsqueeze(0)
            ilens = x.new_tensor([xs.size(1)]).long()
            olens = y.new_tensor([ys.size(1)]).long()
            outs, *_ = self._forward(xs, ilens, ys, olens, speaker_embeddings)

            # get attention weights
            att_ws = []
            for i in range(len(self.decoder.decoders)):
                att_ws += [self.decoder.decoders[i].src_attn.attn]
            att_ws = torch.stack(att_ws, dim=1)  # (B, L, H, T_out, T_in)
            self.train()
            return outs[0], None, att_ws[0]

        # forward encoder
        xs = x.unsqueeze(0)
        hs, _ = self.encoder(xs, None)

        # integrate speaker embedding
        if self.spk_embed_dim is not None:
            speaker_embeddings = speaker_embedding.unsqueeze(0)
            hs = self._integrate_with_spk_embed(hs, speaker_embeddings)

        # set limits of length
        maxlen = int(hs.size(1) * maxlenratio / self.reduction_factor)
        minlen = int(hs.size(1) * minlenratio / self.reduction_factor)

        # initialize
        index = 0
        ys = hs.new_zeros(1, 1, self.odim)
        outs, probs = [], []

        # forward decoder step-by-step
        z_cache = self.decoder.init_state(x)
        while True:
            # update index
            index += 1

            # calculate output and stop prob at index-th step
            y_masks = subsequent_mask(index).unsqueeze(0).to(x.device)
            z, z_cache = self.decoder.forward_one_step(ys, y_masks, hs, cache=z_cache)  # (B, adim)
            outs += [self.feat_out(z).view(self.reduction_factor, self.odim)]  # [(r, odim), ...]
            probs += [torch.sigmoid(self.prob_out(z))[0]]  # [(r), ...]

            # update next inputs
            ys = torch.cat((ys, outs[-1][-1].view(1, 1, self.odim)), dim=1)  # (1, index + 1, odim)

            # get attention weights
            att_ws_ = []
            for name, m in self.named_modules():
                if isinstance(m, MultiHeadedAttention) and "src" in name:
                    att_ws_ += [m.attn[0, :, -1].unsqueeze(1)]  # [(#heads, 1, T),...]
            if index == 1:
                att_ws = att_ws_
            else:
                # [(#heads, l, T), ...]
                att_ws = [torch.cat([att_w, att_w_], dim=1) for att_w, att_w_ in zip(att_ws, att_ws_)]

            # check whether to finish generation
            if int(sum(probs[-1] >= threshold)) > 0 or index >= maxlen:
                # check mininum length
                if index < minlen:
                    continue
                outs = (torch.cat(outs, dim=0).unsqueeze(0).transpose(1, 2))
                # (L, odim) -> (1, L, odim) -> (1, odim, L)
                if self.postnet is not None:
                    outs = outs + self.postnet(outs)  # (1, odim, L)
                outs = outs.transpose(2, 1).squeeze(0)  # (L, odim)
                probs = torch.cat(probs, dim=0)
                break

        # concatenate attention weights -> (#layers, #heads, L, T)
        att_ws = torch.stack(att_ws, dim=0)
        self.train()

        return outs, probs, att_ws

    @staticmethod
    def _add_first_frame_and_remove_last_frame(ys):
        return torch.cat([ys.new_zeros((ys.shape[0], 1, ys.shape[2])), ys[:, :-1]], dim=1)

    def _source_mask(self, ilens):
        """
        Make masks for self-attention.

        Args:
            ilens (LongTensor): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor for self-attention.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)
        """
        x_masks = make_non_pad_mask(ilens).to(ilens.device)
        return x_masks.unsqueeze(-2)

    def _target_mask(self, olens):
        """
        Make masks for masked self-attention.

        Args:
            olens (LongTensor): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor for masked self-attention.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)
        """
        y_masks = make_non_pad_mask(olens).to(olens.device)
        s_masks = subsequent_mask(y_masks.size(-1), device=y_masks.device).unsqueeze(0)
        return y_masks.unsqueeze(-2) & s_masks

    def _integrate_with_spk_embed(self, hs, speaker_embeddings):
        """
        Integrate speaker embedding with hidden states.

        Args:
            hs (Tensor): Batch of hidden state sequences (B, Tmax, adim).
            speaker_embeddings (Tensor): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Tensor: Batch of integrated hidden state sequences (B, Tmax, adim).
        """
        if self.spk_embed_integration_type == "add":
            # apply projection and then add to hidden states
            # speaker_embeddings = F.normalize(speaker_embeddings)
            speaker_embeddings = self.projection(speaker_embeddings)
            hs = hs + speaker_embeddings.unsqueeze(1)
        elif self.spk_embed_integration_type == "concat":
            # concat hidden states with spk embeds and then apply projection
            speaker_embeddings = F.normalize(speaker_embeddings).unsqueeze(1).expand(-1, hs.size(1), -1)
            hs = self.projection(torch.cat([hs, speaker_embeddings], dim=-1))
        else:
            raise NotImplementedError("support only add or concat.")

        return hs
