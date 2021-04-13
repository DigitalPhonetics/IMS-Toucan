import os
from abc import ABC

import numpy as np
import sounddevice
import soundfile
import torch
import torch.nn.functional as F

from Layers.Conformer import Conformer
from Layers.DurationPredictor import DurationPredictor
from Layers.LengthRegulator import LengthRegulator
from Layers.PostNet import PostNet
from Layers.ResidualStack import ResidualStack
from Layers.VariancePredictor import VariancePredictor
from PreprocessingForTTS.ProcessText import TextFrontend
from Utility.utils import make_non_pad_mask
from Utility.utils import make_pad_mask


class FastSpeech2(torch.nn.Module, ABC):

    def __init__(self,  # network structure related
                 idim, odim, adim=384, aheads=4, elayers=6, eunits=1536, dlayers=6, dunits=1536,
                 postnet_layers=5, postnet_chans=256, postnet_filts=5, positionwise_layer_type="conv1d",
                 positionwise_conv_kernel_size=1, use_scaled_pos_enc=True, use_batch_norm=True, encoder_normalize_before=True,
                 decoder_normalize_before=True, encoder_concat_after=False, decoder_concat_after=False, reduction_factor=1,
                 # encoder / decoder
                 conformer_pos_enc_layer_type="rel_pos", conformer_self_attn_layer_type="rel_selfattn", conformer_activation_type="swish",
                 use_macaron_style_in_conformer=True, use_cnn_in_conformer=True, conformer_enc_kernel_size=7,
                 conformer_dec_kernel_size=31,  # duration predictor
                 duration_predictor_layers=2, duration_predictor_chans=256, duration_predictor_kernel_size=3,  # energy predictor
                 energy_predictor_layers=2, energy_predictor_chans=256, energy_predictor_kernel_size=3,
                 energy_predictor_dropout=0.5, energy_embed_kernel_size=1, energy_embed_dropout=0.0,
                 stop_gradient_from_energy_predictor=True,  # pitch predictor
                 pitch_predictor_layers=5, pitch_predictor_chans=256, pitch_predictor_kernel_size=5, pitch_predictor_dropout=0.5,
                 pitch_embed_kernel_size=1, pitch_embed_dropout=0.0, stop_gradient_from_pitch_predictor=True,  # pretrained spk emb
                 spk_embed_dim=None,  # training related
                 transformer_enc_dropout_rate=0.2, transformer_enc_positional_dropout_rate=0.2,
                 transformer_enc_attn_dropout_rate=0.2, transformer_dec_dropout_rate=0.2,
                 transformer_dec_positional_dropout_rate=0.2, transformer_dec_attn_dropout_rate=0.2,
                 duration_predictor_dropout_rate=0.2, postnet_dropout_rate=0.5, init_type="kaiming_uniform",
                 init_enc_alpha=1.0, init_dec_alpha=1.0, use_masking=False, use_weighted_masking=True, lang='en'):
        super().__init__()
        self.idim = idim
        self.odim = odim
        self.reduction_factor = reduction_factor
        self.stop_gradient_from_pitch_predictor = stop_gradient_from_pitch_predictor
        self.stop_gradient_from_energy_predictor = stop_gradient_from_energy_predictor
        self.use_scaled_pos_enc = use_scaled_pos_enc
        self.spk_embed_dim = spk_embed_dim
        self.padding_idx = 0
        encoder_input_layer = torch.nn.Embedding(num_embeddings=idim, embedding_dim=adim, padding_idx=self.padding_idx)
        self.encoder = Conformer(idim=idim, attention_dim=adim, attention_heads=aheads, linear_units=eunits, num_blocks=elayers,
                                 input_layer=encoder_input_layer, dropout_rate=transformer_enc_dropout_rate,
                                 positional_dropout_rate=transformer_enc_positional_dropout_rate, attention_dropout_rate=transformer_enc_attn_dropout_rate,
                                 normalize_before=encoder_normalize_before, concat_after=encoder_concat_after,
                                 positionwise_conv_kernel_size=positionwise_conv_kernel_size, macaron_style=use_macaron_style_in_conformer,
                                 use_cnn_module=use_cnn_in_conformer, cnn_module_kernel=conformer_enc_kernel_size)
        if self.spk_embed_dim is not None:
            self.projection = torch.nn.Linear(adim + self.spk_embed_dim, adim)
        self.duration_predictor = DurationPredictor(idim=adim, n_layers=duration_predictor_layers, n_chans=duration_predictor_chans,
                                                    kernel_size=duration_predictor_kernel_size, dropout_rate=duration_predictor_dropout_rate, )
        self.pitch_predictor = VariancePredictor(idim=adim, n_layers=pitch_predictor_layers, n_chans=pitch_predictor_chans,
                                                 kernel_size=pitch_predictor_kernel_size, dropout_rate=pitch_predictor_dropout)
        self.pitch_embed = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=adim, kernel_size=pitch_embed_kernel_size, padding=(pitch_embed_kernel_size - 1) // 2),
            torch.nn.Dropout(pitch_embed_dropout))
        self.energy_predictor = VariancePredictor(idim=adim, n_layers=energy_predictor_layers, n_chans=energy_predictor_chans,
                                                  kernel_size=energy_predictor_kernel_size, dropout_rate=energy_predictor_dropout)
        self.energy_embed = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=adim, kernel_size=energy_embed_kernel_size, padding=(energy_embed_kernel_size - 1) // 2),
            torch.nn.Dropout(energy_embed_dropout))
        self.length_regulator = LengthRegulator()
        self.decoder = Conformer(idim=0, attention_dim=adim, attention_heads=aheads, linear_units=dunits, num_blocks=dlayers, input_layer=None,
                                 dropout_rate=transformer_dec_dropout_rate, positional_dropout_rate=transformer_dec_positional_dropout_rate,
                                 attention_dropout_rate=transformer_dec_attn_dropout_rate, normalize_before=decoder_normalize_before,
                                 concat_after=decoder_concat_after, positionwise_conv_kernel_size=positionwise_conv_kernel_size,
                                 macaron_style=use_macaron_style_in_conformer, use_cnn_module=use_cnn_in_conformer, cnn_module_kernel=conformer_dec_kernel_size)
        self.feat_out = torch.nn.Linear(adim, odim * reduction_factor)
        self.postnet = PostNet(idim=idim, odim=odim, n_layers=postnet_layers, n_chans=postnet_chans, n_filts=postnet_filts, use_batch_norm=use_batch_norm,
                               dropout_rate=postnet_dropout_rate)
        self.load_state_dict(torch.load(os.path.join("Models", "FastSpeech2_Karlsson", "best.pt"), map_location='cpu')["model"])

    def _forward(self, xs, ilens, ys=None, olens=None, ds=None,
                 ps=None, es=None, spembs=None, is_inference=False, alpha=1.0):
        x_masks = self._source_mask(ilens)
        hs, _ = self.encoder(xs, x_masks)
        if self.spk_embed_dim is not None:
            hs = self._integrate_with_spk_embed(hs, spembs)
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

    def forward(self, text, speech=None, spembs=None, durations=None, pitch=None,
                energy=None, alpha=1.0):
        self.eval()
        x, y = text, speech
        spemb, d, p, e = spembs, durations, pitch, energy
        ilens = torch.tensor([x.shape[0]], dtype=torch.long, device=x.device)
        xs, ys = x.unsqueeze(0), None
        if y is not None:
            ys = y.unsqueeze(0)
        if spemb is not None:
            spembs = spemb.unsqueeze(0)
        _, outs, *_ = self._forward(xs, ilens, ys, spembs=spembs, is_inference=True, alpha=alpha)
        self.train()
        return outs[0]

    def _integrate_with_spk_embed(self, hs, spembs):
        spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
        hs = self.projection(torch.cat([hs, spembs], dim=-1))
        return hs

    def _source_mask(self, ilens):
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        return x_masks.unsqueeze(-2)


class MelGANGenerator(torch.nn.Module):

    def __init__(self, in_channels=80, out_channels=1, kernel_size=7, channels=512, bias=True,
                 upsample_scales=[8, 8, 2, 2], stack_kernel_size=3, stacks=3,
                 nonlinear_activation="LeakyReLU", nonlinear_activation_params={"negative_slope": 0.2},
                 pad="ReflectionPad1d", pad_params={},
                 use_final_nonlinear_activation=True, use_weight_norm=True):
        super(MelGANGenerator, self).__init__()
        assert channels >= np.prod(upsample_scales)
        assert channels % (2 ** len(upsample_scales)) == 0
        assert (kernel_size - 1) % 2 == 0, "even number for kernel size does not work."
        layers = []
        layers += [getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params),
                   torch.nn.Conv1d(in_channels, channels, kernel_size, bias=bias)]
        for i, upsample_scale in enumerate(upsample_scales):
            layers += [getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)]
            layers += [torch.nn.ConvTranspose1d(channels // (2 ** i), channels // (2 ** (i + 1)), upsample_scale * 2, stride=upsample_scale,
                                                padding=upsample_scale // 2 + upsample_scale % 2, output_padding=upsample_scale % 2, bias=bias)]
            for j in range(stacks):
                layers += [ResidualStack(kernel_size=stack_kernel_size, channels=channels // (2 ** (i + 1)), dilation=stack_kernel_size ** j, bias=bias,
                                         nonlinear_activation=nonlinear_activation, nonlinear_activation_params=nonlinear_activation_params, pad=pad,
                                         pad_params=pad_params)]
        layers += [getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)]
        layers += [getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params),
                   torch.nn.Conv1d(channels // (2 ** (i + 1)), out_channels, kernel_size, bias=bias)]

        if use_final_nonlinear_activation:
            layers += [torch.nn.Tanh()]
        self.melgan = torch.nn.Sequential(*layers)
        if use_weight_norm:
            self.apply_weight_norm()
        self.load_state_dict(torch.load(os.path.join("Models", "MelGAN_Karlsson", "best.pt"), map_location='cpu')["generator"])

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


class Karlsson_FastSpeechInference(torch.nn.Module):

    def __init__(self, device="cpu", speaker_embedding=None):
        super().__init__()
        self.device = device
        self.text2phone = TextFrontend(language="de", use_panphon_vectors=False, use_word_boundaries=False, use_explicit_eos=False)
        self.phone2mel = FastSpeech2(idim=133, odim=80, spk_embed_dim=None, reduction_factor=1).to(torch.device(device))
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
            lbd.specshow(mel.cpu().numpy(), ax=ax[1], sr=16000, cmap='GnBu', y_axis='mel', x_axis='time', hop_length=256)
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

    def read_aloud(self, text, view=False, blocking=False):
        if text.strip() == "":
            return

        wav = self(text, view).cpu()

        if not blocking:
            sounddevice.play(wav.numpy(), samplerate=16000)

        else:
            silence = torch.zeros([12000])
            sounddevice.play(torch.cat((wav, silence), 0).numpy(), samplerate=16000)
            sounddevice.wait()
