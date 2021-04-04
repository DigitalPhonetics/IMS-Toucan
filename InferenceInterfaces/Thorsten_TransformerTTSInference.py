import os
from abc import ABC

import numpy as np
import sounddevice
import soundfile
import torch
import torch.nn.functional as F

from Layers.Attention import GuidedMultiHeadAttentionLoss
from Layers.Attention import MultiHeadedAttention
from Layers.PositionalEncoding import PositionalEncoding, ScaledPositionalEncoding
from Layers.PostNet import PostNet
from Layers.ResidualStack import ResidualStack
from Layers.TransformerTTSDecoder import Decoder
from Layers.TransformerTTSDecoderPrenet import DecoderPrenet
from Layers.TransformerTTSEncoder import Encoder
from Layers.TransformerTTSEncoderPrenet import EncoderPrenet
from PreprocessingForTTS.ProcessText import TextFrontend
from TransformerTTS.TransformerLoss import TransformerLoss
from Utility.utils import make_non_pad_mask
from Utility.utils import subsequent_mask


class Transformer(torch.nn.Module, ABC):

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
                 guided_attn_loss_lambda: float = 25.0):
        super().__init__()
        self.idim = idim
        self.odim = odim
        self.eos = idim - 1
        self.spk_embed_dim = spk_embed_dim
        self.reduction_factor = reduction_factor
        self.use_guided_attn_loss = use_guided_attn_loss
        self.use_scaled_pos_enc = use_scaled_pos_enc
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
        self.padding_idx = 0
        pos_enc_class = (ScaledPositionalEncoding if self.use_scaled_pos_enc else PositionalEncoding)
        if eprenet_conv_layers != 0:
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
        if self.spk_embed_dim is not None:
            self.projection = torch.nn.Linear(adim + self.spk_embed_dim, adim)

        decoder_input_layer = torch.nn.Sequential(DecoderPrenet(idim=odim,
                                                                n_layers=dprenet_layers,
                                                                n_units=dprenet_units,
                                                                dropout_rate=dprenet_dropout_rate),
                                                  torch.nn.Linear(dprenet_units, adim))
        self.decoder = Decoder(odim=odim,
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
        self.feat_out = torch.nn.Linear(adim, odim * reduction_factor)
        self.prob_out = torch.nn.Linear(adim, reduction_factor)
        self.postnet = PostNet(idim=idim,
                               odim=odim,
                               n_layers=postnet_layers,
                               n_chans=postnet_chans,
                               n_filts=postnet_filts,
                               use_batch_norm=use_batch_norm,
                               dropout_rate=postnet_dropout_rate)
        if self.use_guided_attn_loss:
            self.attn_criterion = GuidedMultiHeadAttentionLoss(sigma=guided_attn_loss_sigma,
                                                               alpha=guided_attn_loss_lambda)
        self.criterion = TransformerLoss(use_masking=use_masking,
                                         use_weighted_masking=use_weighted_masking,
                                         bce_pos_weight=bce_pos_weight)
        if self.use_guided_attn_loss:
            self.attn_criterion = GuidedMultiHeadAttentionLoss(sigma=guided_attn_loss_sigma,
                                                               alpha=guided_attn_loss_lambda)
        self.load_state_dict(
            torch.load(os.path.join("Models", "TransformerTTS_Thorsten", "best.pt"), map_location='cpu')["model"])

    def forward(self, text: torch.Tensor, spemb=None):
        self.eval()
        x = text
        xs = x.unsqueeze(0)
        hs, _ = self.encoder(xs, None)
        if self.spk_embed_dim is not None:
            spembs = spemb.unsqueeze(0)
            hs = self._integrate_with_spk_embed(hs, spembs)
        maxlen = int(hs.size(1) * 10.0 / self.reduction_factor)
        minlen = int(hs.size(1) * 0.0 / self.reduction_factor)
        idx = 0
        ys = hs.new_zeros(1, 1, self.odim)
        outs, probs = [], []
        z_cache = self.decoder.init_state(x)
        while True:
            idx += 1
            y_masks = subsequent_mask(idx).unsqueeze(0).to(x.device)
            z, z_cache = self.decoder.forward_one_step(ys, y_masks, hs, cache=z_cache)
            outs += [self.feat_out(z).view(self.reduction_factor, self.odim)]
            probs += [torch.sigmoid(self.prob_out(z))[0]]
            ys = torch.cat((ys, outs[-1][-1].view(1, 1, self.odim)), dim=1)
            att_ws_ = []
            for name, m in self.named_modules():
                if isinstance(m, MultiHeadedAttention) and "src" in name:
                    att_ws_ += [m.attn[0, :, -1].unsqueeze(1)]
            if idx == 1:
                att_ws = att_ws_
            else:
                att_ws = [torch.cat([att_w, att_w_], dim=1) for att_w, att_w_ in zip(att_ws, att_ws_)]
            if int(sum(probs[-1] >= 0.5)) > 0 or idx >= maxlen:
                if idx < minlen:
                    continue
                outs = (torch.cat(outs, dim=0).unsqueeze(0).transpose(1, 2))
                if self.postnet is not None:
                    outs = outs + self.postnet(outs)
                outs = outs.transpose(2, 1).squeeze(0)
                break
        return outs

    @staticmethod
    def _add_first_frame_and_remove_last_frame(ys: torch.Tensor):
        return torch.cat([ys.new_zeros((ys.shape[0], 1, ys.shape[2])), ys[:, :-1]], dim=1)

    def _source_mask(self, ilens):
        x_masks = make_non_pad_mask(ilens).to(ilens.device)
        return x_masks.unsqueeze(-2)

    def _target_mask(self, olens: torch.Tensor) -> torch.Tensor:
        y_masks = make_non_pad_mask(olens).to(olens.device)
        s_masks = subsequent_mask(y_masks.size(-1), device=y_masks.device).unsqueeze(0)
        return y_masks.unsqueeze(-2) & s_masks

    def _integrate_with_spk_embed(self, hs: torch.Tensor, spembs: torch.Tensor) -> torch.Tensor:
        spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
        hs = self.projection(torch.cat([hs, spembs], dim=-1))
        return hs


class MelGANGenerator(torch.nn.Module):

    def __init__(self,
                 in_channels=80,
                 out_channels=1,
                 kernel_size=7,
                 channels=512,
                 bias=True,
                 upsample_scales=[8, 4, 2, 2, 2],
                 stack_kernel_size=3,
                 stacks=4,
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.2},
                 pad="ReflectionPad1d",
                 pad_params={},
                 use_final_nonlinear_activation=True,
                 use_weight_norm=True):
        super(MelGANGenerator, self).__init__()
        assert channels >= np.prod(upsample_scales)
        assert channels % (2 ** len(upsample_scales)) == 0
        assert (kernel_size - 1) % 2 == 0, "even number for kernel size does not work."
        layers = []
        layers += [getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params),
                   torch.nn.Conv1d(in_channels, channels, kernel_size, bias=bias)]
        for i, upsample_scale in enumerate(upsample_scales):
            layers += [getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)]
            layers += [torch.nn.ConvTranspose1d(channels // (2 ** i),
                                                channels // (2 ** (i + 1)),
                                                upsample_scale * 2,
                                                stride=upsample_scale,
                                                padding=upsample_scale // 2 + upsample_scale % 2,
                                                output_padding=upsample_scale % 2,
                                                bias=bias)]
            for j in range(stacks):
                layers += [ResidualStack(kernel_size=stack_kernel_size,
                                         channels=channels // (2 ** (i + 1)),
                                         dilation=stack_kernel_size ** j,
                                         bias=bias,
                                         nonlinear_activation=nonlinear_activation,
                                         nonlinear_activation_params=nonlinear_activation_params,
                                         pad=pad,
                                         pad_params=pad_params)]
        layers += [getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)]
        layers += [getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params),
                   torch.nn.Conv1d(channels // (2 ** (i + 1)), out_channels, kernel_size, bias=bias)]

        if use_final_nonlinear_activation:
            layers += [torch.nn.Tanh()]
        self.melgan = torch.nn.Sequential(*layers)
        if use_weight_norm:
            self.apply_weight_norm()
        self.load_state_dict(
            torch.load(os.path.join("Models", "MelGAN_Thorsten", "best.pt"), map_location='cpu')["generator"])

    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def forward(self, melspec):
        self.melgan.eval()
        return self.melgan(melspec)


class Thorsten_TransformerTTSInference(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.text2phone = TextFrontend(language="de",
                                       use_panphon_vectors=False,
                                       use_word_boundaries=False,
                                       use_explicit_eos=False)
        self.phone2mel = Transformer(idim=133, odim=80, spk_embed_dim=None,
                                     reduction_factor=1).to(torch.device(device))
        self.mel2wav = MelGANGenerator().to(torch.device(device))
        self.phone2mel.eval()
        self.mel2wav.eval()
        self.to(torch.device(device))

    def forward(self, text, view=False):
        with torch.no_grad():
            phones = self.text2phone.string_to_tensor(text).squeeze(0).long().to(torch.device(self.device))
            mel = self.phone2mel(phones).transpose(0, 1)
            wave = self.mel2wav(mel.unsqueeze(0)).squeeze(0).squeeze(0)
        if view:
            import matplotlib.pyplot as plt
            import librosa.display as lbd
            fig, ax = plt.subplots(nrows=2, ncols=1)
            ax[0].plot(wave.cpu().numpy())
            lbd.specshow(mel.cpu().numpy(), ax=ax[1], sr=16000, cmap='GnBu', y_axis='mel', x_axis='time',
                         hop_length=256)
            ax[0].set_title(self.text2phone.get_phone_string(text))
            ax[0].yaxis.set_visible(False)
            ax[1].yaxis.set_visible(False)
            plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=.9, wspace=0.0, hspace=0.0)
            plt.show()

        return wave

    def read_to_file(self, text_list, file_location, silent=False):
        """
        :param silent: Whether to be verbose about the process
        :param text_list: A list of strings to be read
        :param file_location: The path and name of the file it should be saved to
        """
        wav = None
        silence = torch.zeros([8000])
        for text in text_list:
            if text.strip() != "":
                if not silent:
                    print("Now synthesizing: {}".format(text))
                if wav is None:
                    wav = self(text).cpu()
                else:
                    wav = torch.cat((wav, silence), 0)
                    wav = torch.cat((wav, self(text).cpu()), 0)
        soundfile.write(file=file_location, data=wav.cpu().numpy(), samplerate=16000)

    def read_aloud(self, text, view=False):
        wav = self(text, view).cpu().numpy()
        sounddevice.play(wav, samplerate=16000)
