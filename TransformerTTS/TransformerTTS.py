# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""TTS-Transformer related modules."""
import os
from abc import ABC
from typing import Dict

import torch
import torch.nn.functional as F

from Layers.Attention import GuidedMultiHeadAttentionLoss
from Layers.Attention import MultiHeadedAttention
from Layers.PositionalEncoding import PositionalEncoding, ScaledPositionalEncoding
from Layers.PostNet import PostNet
from Layers.TransformerTTSDecoder import Decoder
from Layers.TransformerTTSDecoderPrenet import DecoderPrenet
from Layers.TransformerTTSEncoder import Encoder
from Layers.TransformerTTSEncoderPrenet import EncoderPrenet
from TransformerTTS.TransformerLoss import TransformerLoss
from Utility.utils import make_pad_mask, make_non_pad_mask, initialize
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
                 idim: int,
                 odim: int,
                 embed_dim: int = 0,
                 eprenet_conv_layers: int = 0,
                 eprenet_conv_chans: int = 0,
                 eprenet_conv_filts: int = 0,
                 dprenet_layers: int = 2,
                 dprenet_units: int = 256,
                 elayers: int = 6,
                 eunits: int = 1024,
                 adim: int = 512,
                 aheads: int = 4,
                 dlayers: int = 6,
                 dunits: int = 1024,
                 postnet_layers: int = 5,
                 postnet_chans: int = 256,
                 postnet_filts: int = 5,
                 positionwise_layer_type: str = "conv1d",
                 positionwise_conv_kernel_size: int = 1,
                 use_scaled_pos_enc: bool = True,
                 use_batch_norm: bool = True,
                 encoder_normalize_before: bool = True,
                 decoder_normalize_before: bool = True,
                 encoder_concat_after: bool = True,  # True according to https://github.com/soobinseo/Transformer-TTS
                 decoder_concat_after: bool = True,  # True according to https://github.com/soobinseo/Transformer-TTS
                 reduction_factor=1,
                 spk_embed_dim: int = None,
                 spk_embed_integration_type: str = "concat",
                 # training related
                 transformer_enc_dropout_rate: float = 0.1,
                 transformer_enc_positional_dropout_rate: float = 0.1,
                 transformer_enc_attn_dropout_rate: float = 0.1,
                 transformer_dec_dropout_rate: float = 0.1,
                 transformer_dec_positional_dropout_rate: float = 0.1,
                 transformer_dec_attn_dropout_rate: float = 0.1,
                 transformer_enc_dec_attn_dropout_rate: float = 0.1,
                 eprenet_dropout_rate: float = 0.0,
                 dprenet_dropout_rate: float = 0.5,
                 postnet_dropout_rate: float = 0.5,
                 init_type: str = "xavier_uniform",  # since we have little to no
                 # asymetric activations, this seems to work better than kaiming
                 init_enc_alpha: float = 1.0,
                 use_masking: bool = False,  # either this or weighted masking, not both
                 use_weighted_masking: bool = True,  # if there are severely different sized samples in one batch
                 bce_pos_weight: float = 7.0,  # scaling the loss of the stop token prediction
                 loss_type: str = "L1",
                 use_guided_attn_loss: bool = True,
                 num_heads_applied_guided_attn: int = 2,
                 num_layers_applied_guided_attn: int = 2,
                 modules_applied_guided_attn=("encoder-decoder",),
                 guided_attn_loss_sigma: float = 0.4,  # standard deviation from diagonal that is allowed
                 guided_attn_loss_lambda: float = 25.0):  # forcing the attention to be diagonal
        """Initialize Transformer module."""
        super().__init__()

        # store hyperparameters
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
            encoder_input_layer = torch.nn.Sequential(EncoderPrenet(idim=idim,
                                                                    embed_dim=embed_dim,
                                                                    elayers=0,
                                                                    econv_layers=eprenet_conv_layers,
                                                                    econv_chans=eprenet_conv_chans,
                                                                    econv_filts=eprenet_conv_filts,
                                                                    use_batch_norm=use_batch_norm,
                                                                    dropout_rate=eprenet_dropout_rate,
                                                                    padding_idx=self.padding_idx),
                                                      torch.nn.Linear(eprenet_conv_chans, adim))
        else:
            encoder_input_layer = torch.nn.Embedding(num_embeddings=idim, embedding_dim=adim,
                                                     padding_idx=self.padding_idx)
        self.encoder = Encoder(idim=idim,
                               attention_dim=adim,
                               attention_heads=aheads,
                               linear_units=eunits,
                               num_blocks=elayers,
                               input_layer=encoder_input_layer,
                               dropout_rate=transformer_enc_dropout_rate,
                               positional_dropout_rate=transformer_enc_positional_dropout_rate,
                               attention_dropout_rate=transformer_enc_attn_dropout_rate,
                               pos_enc_class=pos_enc_class,
                               normalize_before=encoder_normalize_before,
                               concat_after=encoder_concat_after,
                               positionwise_layer_type=positionwise_layer_type,
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
            decoder_input_layer = torch.nn.Sequential(DecoderPrenet(idim=odim,
                                                                    n_layers=dprenet_layers,
                                                                    n_units=dprenet_units,
                                                                    dropout_rate=dprenet_dropout_rate),
                                                      torch.nn.Linear(dprenet_units, adim))
        else:
            decoder_input_layer = "linear"
        self.decoder = Decoder(odim=odim,  # odim is needed when no prenet is used
                               attention_dim=adim,
                               attention_heads=aheads,
                               linear_units=dunits,
                               num_blocks=dlayers,
                               dropout_rate=transformer_dec_dropout_rate,
                               positional_dropout_rate=transformer_dec_positional_dropout_rate,
                               self_attention_dropout_rate=transformer_dec_attn_dropout_rate,
                               src_attention_dropout_rate=transformer_enc_dec_attn_dropout_rate,
                               input_layer=decoder_input_layer,
                               use_output_layer=False,
                               pos_enc_class=pos_enc_class,
                               normalize_before=decoder_normalize_before,
                               concat_after=decoder_concat_after)

        # define final projection
        self.feat_out = torch.nn.Linear(adim, odim * reduction_factor)
        self.prob_out = torch.nn.Linear(adim, reduction_factor)

        # define postnet
        self.postnet = PostNet(idim=idim,
                               odim=odim,
                               n_layers=postnet_layers,
                               n_chans=postnet_chans,
                               n_filts=postnet_filts,
                               use_batch_norm=use_batch_norm,
                               dropout_rate=postnet_dropout_rate)

        # define loss function
        self.criterion = TransformerLoss(use_masking=use_masking,
                                         use_weighted_masking=use_weighted_masking,
                                         bce_pos_weight=bce_pos_weight)
        if self.use_guided_attn_loss:
            self.attn_criterion = GuidedMultiHeadAttentionLoss(sigma=guided_attn_loss_sigma,
                                                               alpha=guided_attn_loss_lambda)

        # initialize parameters
        self._reset_parameters(init_type=init_type,
                               init_enc_alpha=init_enc_alpha,
                               init_dec_alpha=init_enc_alpha)

    def _reset_parameters(self, init_type, init_enc_alpha=1.0, init_dec_alpha=1.0):
        # initialize parameters
        if init_type != "pytorch":
            initialize(self, init_type)

        # initialize alpha in scaled positional encoding
        if self.use_scaled_pos_enc:
            self.encoder.embed[-1].alpha.data = torch.tensor(init_enc_alpha)
            self.decoder.embed[-1].alpha.data = torch.tensor(init_dec_alpha)

    def forward(self,
                text: torch.Tensor,
                text_lengths: torch.Tensor,
                speech: torch.Tensor,
                speech_lengths: torch.Tensor,
                spembs: torch.Tensor = None):
        """Calculate forward propagation.

        Args:
            text (LongTensor): Batch of padded character ids (B, Tmax).
            text_lengths (LongTensor): Batch of lengths of each input batch (B,).
            speech (Tensor): Batch of padded target features (B, Lmax, odim).
            speech_lengths (LongTensor): Batch of the lengths of each target (B,).
            spembs (Tensor, optional): Batch of speaker embeddings (B, spk_embed_dim).

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
        after_outs, before_outs, logits = self._forward(text, text_lengths, speech, speech_lengths, spembs)

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

    def _forward(self,
                 xs: torch.Tensor,
                 ilens: torch.Tensor,
                 ys: torch.Tensor,
                 olens: torch.Tensor,
                 spembs: torch.Tensor):
        # forward encoder
        x_masks = self._source_mask(ilens)
        hs, h_masks = self.encoder(xs, x_masks)

        # integrate speaker embedding
        if self.spk_embed_dim is not None:
            hs = self._integrate_with_spk_embed(hs, spembs)

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

    def inference(self,
                  text: torch.Tensor,
                  speech: torch.Tensor = None,
                  spembs: torch.Tensor = None,
                  threshold: float = 0.5,
                  minlenratio: float = 0.0,
                  maxlenratio: float = 10.0,
                  use_teacher_forcing: bool = False):
        """
        Generate the sequence of features given the sequences of characters.

        Args:
            text (LongTensor): Input sequence of characters (T,).
            speech (Tensor, optional): Feature sequence to extract style (N, idim).
            spembs (Tensor, optional): Speaker embedding vector (spk_embed_dim,).
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
        spemb = spembs
        self.eval()

        # add eos at the last of sequence
        x = F.pad(x, [0, 1], "constant", self.eos)

        # inference with teacher forcing
        if use_teacher_forcing:
            assert speech is not None, "speech must be provided with teacher forcing."

            # get teacher forcing outputs
            xs, ys = x.unsqueeze(0), y.unsqueeze(0)
            spembs = None if spemb is None else spemb.unsqueeze(0)
            ilens = x.new_tensor([xs.size(1)]).long()
            olens = y.new_tensor([ys.size(1)]).long()
            outs, *_ = self._forward(xs, ilens, ys, olens, spembs)

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
            spembs = spemb.unsqueeze(0)
            hs = self._integrate_with_spk_embed(hs, spembs)

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
    def _add_first_frame_and_remove_last_frame(ys: torch.Tensor):
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

    def _target_mask(self, olens: torch.Tensor) -> torch.Tensor:
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

    def _integrate_with_spk_embed(self, hs: torch.Tensor, spembs: torch.Tensor) -> torch.Tensor:
        """
        Integrate speaker embedding with hidden states.

        Args:
            hs (Tensor): Batch of hidden state sequences (B, Tmax, adim).
            spembs (Tensor): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Tensor: Batch of integrated hidden state sequences (B, Tmax, adim).
        """
        if self.spk_embed_integration_type == "add":
            # apply projection and then add to hidden states
            # spembs = F.normalize(spembs)
            spembs = self.projection(spembs)
            hs = hs + spembs.unsqueeze(1)
        elif self.spk_embed_integration_type == "concat":
            # concat hidden states with spk embeds and then apply projection
            spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
            hs = self.projection(torch.cat([hs, spembs], dim=-1))
        else:
            raise NotImplementedError("support only add or concat.")

        return hs

    def get_conf(self):
        return "idim:{}\n" \
               "odim:{}\n" \
               "spk_embed_dim:{}\n" \
               "reduction_factor:{}\n" \
               "aheads:{}\n" \
               "adim:{}".format(self.idim,
                                self.odim,
                                self.spk_embed_dim,
                                self.reduction_factor,
                                self.aheads,
                                self.adim)


def build_reference_transformer_tts_model(model_name="Transformer_German_Single.pt"):
    model = Transformer(idim=133, odim=80, spk_embed_dim=None).to("cpu")
    params = torch.load(os.path.join("Models", "Use", model_name), map_location='cpu')["model"]
    model.load_state_dict(params)
    return model


def show_spectrogram(sentence, model=None, lang="en"):
    if model is None:
        if lang == "en":
            model = build_reference_transformer_tts_model(model_name="Transformer_English_Single.pt")
        elif lang == "de":
            model = build_reference_transformer_tts_model(model_name="Transformer_German_Single.pt")
    from PreprocessingForTTS.ProcessText import TextFrontend
    import librosa.display as lbd
    import matplotlib.pyplot as plt
    tf = TextFrontend(language=lang,
                      use_panphon_vectors=False,
                      use_word_boundaries=False,
                      use_explicit_eos=False)
    fig, ax = plt.subplots()
    ax.set(title=sentence)
    melspec = model.inference(tf.string_to_tensor(sentence).squeeze(0).long())[0]
    lbd.specshow(melspec.transpose(0, 1).detach().numpy(), ax=ax, sr=16000, cmap='GnBu', y_axis='mel',
                 x_axis='time', hop_length=256)
    plt.show()


def select_best_att_head(att_ws):
    att_ws = torch.cat([att_w for att_w in att_ws], dim=0)
    diagonal_scores = att_ws.max(dim=-1)[0].mean(dim=-1)
    diagonal_head_idx = diagonal_scores.argmax()
    att_ws = att_ws[diagonal_head_idx]
    return att_ws


def plot_attention(att, sentence=None, phones=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.imshow(att.detach().numpy(), interpolation='nearest', aspect='auto', origin="lower")
    if phones is not None:
        plt.xticks(range(len(att[0])), labels=[phone for phone in phones])
    plt.xlabel("Inputs")
    plt.ylabel("Outputs")
    if sentence is not None:
        plt.title(sentence)
    plt.tight_layout()
    plt.show()


def plot_attentions(atts):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows=len(atts) // 2, ncols=2, figsize=(6, 8))
    atts_1 = atts[::2]
    atts_2 = atts[1::2]
    for index, att in enumerate(atts_1):
        axes[index][0].imshow(att.detach().numpy(), interpolation='nearest', aspect='auto',
                              origin="lower")
        axes[index][0].xaxis.set_visible(False)
        axes[index][0].yaxis.set_visible(False)
    for index, att in enumerate(atts_2):
        axes[index][1].imshow(att.detach().numpy(), interpolation='nearest', aspect='auto',
                              origin="lower")
        axes[index][1].xaxis.set_visible(False)
        axes[index][1].yaxis.set_visible(False)
    plt.subplots_adjust(left=0.02, bottom=0.02, right=.98, top=.98, wspace=0, hspace=0)
    plt.show()


def get_atts(model, sentence, lang, teacher_forcing, get_phones=False):
    from PreprocessingForTTS.ProcessText import TextFrontend
    tf = TextFrontend(language=lang,
                      use_panphon_vectors=False,
                      use_word_boundaries=False,
                      use_explicit_eos=False)
    if teacher_forcing:
        from PreprocessingForTTS.ProcessAudio import AudioPreprocessor
        import soundfile as sf
        wave, sr = sf.read("Corpora/att_align_test.wav")
        ap = AudioPreprocessor(input_sr=sr, output_sr=16000)
        spec = ap.audio_to_mel_spec_tensor(wave).transpose(0, 1)
        sentence = "Many animals of even complex structure which " \
                   "live parasitically within others are wholly " \
                   "devoid of an alimentary cavity."
        text_tensor = tf.string_to_tensor(sentence).squeeze(0).long()
        phones = tf.get_phone_string(sentence)
        if get_phones:
            return model.inference(text=text_tensor, speech=spec, use_teacher_forcing=True)[2], phones
        return model.inference(text=text_tensor, speech=spec, use_teacher_forcing=True)[2]
    else:
        if get_phones:
            return model.inference(tf.string_to_tensor(sentence).squeeze(0).long())[2], tf.get_phone_string(sentence)
        return model.inference(tf.string_to_tensor(sentence).squeeze(0).long())[2]


def show_attention_plot(sentence, model=None, best_only=False, lang="en", teacher_forcing=False):
    if model is None:
        if lang == "en":
            model = build_reference_transformer_tts_model(model_name="Transformer_English_Single.pt")
        elif lang == "de":
            model = build_reference_transformer_tts_model(model_name="Transformer_German_Single.pt")

    if best_only:
        att, phones = get_atts(model=model, sentence=sentence, lang=lang, teacher_forcing=teacher_forcing,
                               get_phones=True)
        plot_attention(select_best_att_head(att), sentence=sentence, phones=phones)
    else:
        att, phones = get_atts(model=model, sentence=sentence, lang=lang, teacher_forcing=teacher_forcing,
                               get_phones=True)
        atts = torch.cat(
            [att_w for att_w in att],
            dim=0)
        plot_attentions(atts)
