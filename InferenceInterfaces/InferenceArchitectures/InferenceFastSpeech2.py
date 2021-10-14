from abc import ABC

import torch

from Layers.Conformer import Conformer
from Layers.DurationPredictor import DurationPredictor
from Layers.LengthRegulator import LengthRegulator
from Layers.PostNet import PostNet
from Layers.VariancePredictor import VariancePredictor
from Utility.utils import make_non_pad_mask
from Utility.utils import make_pad_mask


class FastSpeech2(torch.nn.Module, ABC):

    def __init__(self,  # network structure related
                 path_to_weights,
                 idim=66,
                 odim=80,
                 adim=384,
                 aheads=4,
                 elayers=6,
                 eunits=1536,
                 dlayers=6,
                 dunits=1536,
                 postnet_layers=5,
                 postnet_chans=256,
                 postnet_filts=5,
                 positionwise_conv_kernel_size=1,
                 use_scaled_pos_enc=True,
                 use_batch_norm=True,
                 encoder_normalize_before=True,
                 decoder_normalize_before=True,
                 encoder_concat_after=False,
                 decoder_concat_after=False,
                 reduction_factor=1,
                 # encoder / decoder
                 use_macaron_style_in_conformer=True,
                 use_cnn_in_conformer=True,
                 conformer_enc_kernel_size=7,
                 conformer_dec_kernel_size=31,
                 # duration predictor
                 duration_predictor_layers=2,
                 duration_predictor_chans=256,
                 duration_predictor_kernel_size=3,
                 # energy predictor
                 energy_predictor_layers=2,
                 energy_predictor_chans=256,
                 energy_predictor_kernel_size=3,
                 energy_predictor_dropout=0.5,
                 energy_embed_kernel_size=1,
                 energy_embed_dropout=0.0,
                 stop_gradient_from_energy_predictor=True,
                 # pitch predictor
                 pitch_predictor_layers=5,
                 pitch_predictor_chans=256,
                 pitch_predictor_kernel_size=5,
                 pitch_predictor_dropout=0.5,
                 pitch_embed_kernel_size=1,
                 pitch_embed_dropout=0.0,
                 stop_gradient_from_pitch_predictor=True,
                 # training related
                 transformer_enc_dropout_rate=0.2,
                 transformer_enc_positional_dropout_rate=0.2,
                 transformer_enc_attn_dropout_rate=0.2,
                 transformer_dec_dropout_rate=0.2,
                 transformer_dec_positional_dropout_rate=0.2,
                 transformer_dec_attn_dropout_rate=0.2,
                 duration_predictor_dropout_rate=0.2,
                 postnet_dropout_rate=0.5):
        super().__init__()
        self.idim = idim
        self.odim = odim
        self.reduction_factor = reduction_factor
        self.stop_gradient_from_pitch_predictor = stop_gradient_from_pitch_predictor
        self.stop_gradient_from_energy_predictor = stop_gradient_from_energy_predictor
        self.use_scaled_pos_enc = use_scaled_pos_enc
        embed = torch.nn.Sequential(torch.nn.Linear(idim, 100),
                                    torch.nn.Tanh(),
                                    torch.nn.Linear(100, adim))
        self.encoder = Conformer(idim=idim,
                                 attention_dim=adim,
                                 attention_heads=aheads,
                                 linear_units=eunits,
                                 num_blocks=elayers,
                                 input_layer=embed,
                                 dropout_rate=transformer_enc_dropout_rate,
                                 positional_dropout_rate=transformer_enc_positional_dropout_rate,
                                 attention_dropout_rate=transformer_enc_attn_dropout_rate,
                                 normalize_before=encoder_normalize_before,
                                 concat_after=encoder_concat_after,
                                 positionwise_conv_kernel_size=positionwise_conv_kernel_size,
                                 macaron_style=use_macaron_style_in_conformer,
                                 use_cnn_module=use_cnn_in_conformer,
                                 cnn_module_kernel=conformer_enc_kernel_size)
        self.duration_predictor = DurationPredictor(idim=adim, n_layers=duration_predictor_layers,
                                                    n_chans=duration_predictor_chans,
                                                    kernel_size=duration_predictor_kernel_size,
                                                    dropout_rate=duration_predictor_dropout_rate, )
        self.pitch_predictor = VariancePredictor(idim=adim, n_layers=pitch_predictor_layers,
                                                 n_chans=pitch_predictor_chans,
                                                 kernel_size=pitch_predictor_kernel_size,
                                                 dropout_rate=pitch_predictor_dropout)
        self.pitch_embed = torch.nn.Sequential(torch.nn.Conv1d(in_channels=1, out_channels=adim,
                                                               kernel_size=pitch_embed_kernel_size,
                                                               padding=(pitch_embed_kernel_size - 1) // 2),
                                               torch.nn.Dropout(pitch_embed_dropout))
        self.energy_predictor = VariancePredictor(idim=adim, n_layers=energy_predictor_layers,
                                                  n_chans=energy_predictor_chans,
                                                  kernel_size=energy_predictor_kernel_size,
                                                  dropout_rate=energy_predictor_dropout)
        self.energy_embed = torch.nn.Sequential(torch.nn.Conv1d(in_channels=1, out_channels=adim,
                                                                kernel_size=energy_embed_kernel_size,
                                                                padding=(energy_embed_kernel_size - 1) // 2),
                                                torch.nn.Dropout(energy_embed_dropout))
        self.length_regulator = LengthRegulator()
        self.decoder = Conformer(idim=0,
                                 attention_dim=adim,
                                 attention_heads=aheads,
                                 linear_units=dunits,
                                 num_blocks=dlayers,
                                 input_layer=None,
                                 dropout_rate=transformer_dec_dropout_rate,
                                 positional_dropout_rate=transformer_dec_positional_dropout_rate,
                                 attention_dropout_rate=transformer_dec_attn_dropout_rate,
                                 normalize_before=decoder_normalize_before,
                                 concat_after=decoder_concat_after,
                                 positionwise_conv_kernel_size=positionwise_conv_kernel_size,
                                 macaron_style=use_macaron_style_in_conformer,
                                 use_cnn_module=use_cnn_in_conformer,
                                 cnn_module_kernel=conformer_dec_kernel_size)
        self.feat_out = torch.nn.Linear(adim, odim * reduction_factor)
        self.postnet = PostNet(idim=idim,
                               odim=odim,
                               n_layers=postnet_layers,
                               n_chans=postnet_chans,
                               n_filts=postnet_filts,
                               use_batch_norm=use_batch_norm,
                               dropout_rate=postnet_dropout_rate)
        self.load_state_dict(torch.load(path_to_weights, map_location='cpu')["model"])

    def _forward(self,
                 xs,
                 ilens,
                 ys=None,
                 olens=None,
                 ds=None,
                 ps=None,
                 es=None,
                 is_inference=False,
                 alpha=1.0):
        x_masks = self._source_mask(ilens)
        hs, _ = self.encoder(xs, x_masks)

        d_masks = make_pad_mask(ilens).to(xs.device)
        if self.stop_gradient_from_pitch_predictor:
            p_outs = self.pitch_predictor(hs.detach(), d_masks.unsqueeze(-1))
        else:
            p_outs = self.pitch_predictor(hs, d_masks.unsqueeze(-1))
        if self.stop_gradient_from_energy_predictor:
            e_outs = self.energy_predictor(hs.detach(), d_masks.unsqueeze(-1))
        else:
            e_outs = self.energy_predictor(hs, d_masks.unsqueeze(-1))
        if is_inference:
            d_outs = self.duration_predictor.inference(hs, d_masks)
            p_embs = self.pitch_embed(p_outs.transpose(1, 2)).transpose(1, 2)
            e_embs = self.energy_embed(e_outs.transpose(1, 2)).transpose(1, 2)
            hs = hs + e_embs + p_embs
            hs = self.length_regulator(hs, d_outs, alpha)
        else:
            d_outs = self.duration_predictor(hs, d_masks)
            p_embs = self.pitch_embed(ps.transpose(1, 2)).transpose(1, 2)
            e_embs = self.energy_embed(es.transpose(1, 2)).transpose(1, 2)
            hs = hs + e_embs + p_embs
            hs = self.length_regulator(hs, ds)
        if olens is not None and not is_inference:
            if self.reduction_factor > 1:
                olens_in = olens.new([olen // self.reduction_factor for olen in olens])
            else:
                olens_in = olens
            h_masks = self._source_mask(olens_in)
        else:
            h_masks = None
        zs, _ = self.decoder(hs, h_masks)
        before_outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)
        after_outs = before_outs + self.postnet(before_outs.transpose(1, 2)).transpose(1, 2)
        return before_outs, after_outs, d_outs, p_outs, e_outs

    def forward(self, text, alpha=1.0, return_duration_pitch_energy=False):
        self.eval()
        x = text
        ilens = torch.tensor([x.shape[0]], dtype=torch.long, device=x.device)
        xs = x.unsqueeze(0)

        before_outs, after_outs, d_outs, pitch_predictions, energy_predictions = self._forward(xs,
                                                                                               ilens,
                                                                                               None,
                                                                                               is_inference=True,
                                                                                               alpha=alpha)
        self.train()
        if return_duration_pitch_energy:
            return after_outs[0], d_outs[0], pitch_predictions[0], energy_predictions[0]
        return after_outs[0]

    def _source_mask(self, ilens):
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        return x_masks.unsqueeze(-2)
