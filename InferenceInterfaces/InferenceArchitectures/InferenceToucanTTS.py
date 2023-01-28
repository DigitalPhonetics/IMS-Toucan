import torch
import torch.distributions as dist

from Layers.Conformer import Conformer
from Layers.LengthRegulator import LengthRegulator
from Preprocessing.articulatory_features import get_feature_to_index_lookup
from TrainingInterfaces.Text_to_Spectrogram.PortaSpeech.FVAE import FVAE
from TrainingInterfaces.Text_to_Spectrogram.PortaSpeech.Glow import Glow
from TrainingInterfaces.Text_to_Spectrogram.PortaSpeech.PortaSpeechLayers import ConvBlocks
from Utility.utils import make_estimated_durations_usable_for_inference
from Utility.utils import make_non_pad_mask


class PortaSpeech(torch.nn.Module):

    def __init__(self,  # network structure related
                 weights,
                 idim=62,
                 odim=80,
                 adim=192,
                 aheads=4,
                 elayers=6,
                 eunits=1536,
                 positionwise_conv_kernel_size=1,
                 use_scaled_pos_enc=True,
                 encoder_normalize_before=True,
                 encoder_concat_after=False,
                 # encoder / decoder
                 use_macaron_style_in_conformer=True,
                 use_cnn_in_conformer=True,
                 conformer_enc_kernel_size=7,
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
                 duration_predictor_dropout_rate=0.2,
                 # additional features
                 utt_embed_dim=256,
                 lang_embs=8000):
        super().__init__()
        self.idim = idim
        self.odim = odim
        self.adim = adim
        self.stop_gradient_from_pitch_predictor = stop_gradient_from_pitch_predictor
        self.stop_gradient_from_energy_predictor = stop_gradient_from_energy_predictor
        self.use_scaled_pos_enc = use_scaled_pos_enc
        self.multilingual_model = lang_embs is not None
        self.multispeaker_model = utt_embed_dim is not None

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
                                 cnn_module_kernel=conformer_enc_kernel_size,
                                 zero_triu=False,
                                 utt_embed=utt_embed_dim,
                                 lang_embs=lang_embs)
        # define duration predictor
        self.duration_vae = FVAE(c_in=1,  # 1 dimensional random variable based sequence
                                 c_out=1,  # 1 dimensional output sequence
                                 hidden_size=adim // 2,  # size of embedding space in which the processing happens
                                 c_latent=adim // 12,  # latent space inbetween encoder and decoder
                                 kernel_size=duration_predictor_kernel_size,
                                 enc_n_layers=duration_predictor_layers,
                                 dec_n_layers=duration_predictor_layers,
                                 c_cond=adim,  # condition to guide the sampling
                                 strides=[1],
                                 use_prior_flow=False,
                                 flow_hidden=None,
                                 flow_kernel_size=None,
                                 flow_n_steps=None,
                                 norm_type="cln" if utt_embed_dim is not None else "ln",
                                 spk_emb_size=utt_embed_dim)

        # define pitch predictor
        self.pitch_vae = FVAE(c_in=1,
                              c_out=1,
                              hidden_size=adim // 2,
                              c_latent=adim // 12,
                              kernel_size=pitch_predictor_kernel_size,
                              enc_n_layers=pitch_predictor_layers,
                              dec_n_layers=pitch_predictor_layers,
                              c_cond=adim,
                              strides=[1],
                              use_prior_flow=False,
                              flow_hidden=None,
                              flow_kernel_size=None,
                              flow_n_steps=None,
                              norm_type="cln" if utt_embed_dim is not None else "ln",
                              spk_emb_size=utt_embed_dim)
        self.pitch_embed = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1,
                            out_channels=adim,
                            kernel_size=pitch_embed_kernel_size,
                            padding=(pitch_embed_kernel_size - 1) // 2),
            torch.nn.Dropout(pitch_embed_dropout))

        # define energy predictor
        self.energy_vae = FVAE(c_in=1,
                               c_out=1,
                               hidden_size=adim // 2,
                               c_latent=adim // 12,
                               kernel_size=energy_predictor_kernel_size,
                               enc_n_layers=energy_predictor_layers,
                               dec_n_layers=energy_predictor_layers,
                               c_cond=adim,
                               strides=[1],
                               use_prior_flow=False,
                               flow_hidden=None,
                               flow_kernel_size=None,
                               flow_n_steps=None,
                               norm_type="cln" if utt_embed_dim is not None else "ln",
                               spk_emb_size=utt_embed_dim)
        self.energy_embed = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1,
                            out_channels=adim,
                            kernel_size=energy_embed_kernel_size,
                            padding=(energy_embed_kernel_size - 1) // 2),
            torch.nn.Dropout(energy_embed_dropout))

        # define length regulator
        self.length_regulator = LengthRegulator()

        # decoder is just a bunch of conv blocks, the postnet does the heavy lifting.
        # It's not perfect, but with the pitch and energy embeddings, as well as the
        # explicit durations, we don't really need that much expressive power in the decoding.
        self.decoder = ConvBlocks(
            hidden_size=adim,
            out_dims=adim,
            dilations=[1] * 8,
            kernel_size=5,
            norm_type='cln' if utt_embed_dim is not None else 'ln',
            layers_in_block=2,
            c_multiple=2,
            dropout=0.3,
            ln_eps=1e-5,
            init_weights=False,
            is_BTC=True,
            num_layers=None,
            post_net_kernel=3
        )
        self.out_proj = torch.nn.Conv1d(adim, odim, 1)

        # post net is realized as a flow
        gin_channels = adim
        self.post_flow = Glow(
            odim,
            192,  # post_glow_hidden  (original 192 in paper)
            3,  # post_glow_kernel_size
            1,
            12,  # post_glow_n_blocks
            3,  # post_glow_n_block_layers
            n_split=4,
            n_sqz=2,
            gin_channels=gin_channels,
            share_cond_layers=False,  # post_share_cond_layers
            share_wn_layers=4,  # share_wn_layers
            sigmoid_scale=False  # sigmoid_scale
        )
        self.prior_dist = dist.Normal(0, 1)

        self.g_proj = torch.nn.Conv1d(odim + adim, gin_channels, 5, padding=2)

        self.load_state_dict(weights)
        self.eval()

    def _forward(self,
                 text_tensors,
                 text_lens,
                 gold_durations=None,
                 gold_pitch=None,
                 gold_energy=None,
                 duration_scaling_factor=1.0,
                 utterance_embedding=None,
                 lang_ids=None,
                 pitch_variance_scale=1.0,
                 energy_variance_scale=1.0,
                 pause_duration_scaling_factor=1.0,
                 device=None):

        if not self.multilingual_model:
            lang_ids = None

        if not self.multispeaker_model:
            utterance_embedding = None

        # forward encoder
        text_masks = self._source_mask(text_lens)

        encoded_texts, _ = self.encoder(text_tensors,
                                        text_masks,
                                        utterance_embedding=utterance_embedding,
                                        lang_ids=lang_ids)  # (B, Tmax, adim)

        # forward duration predictor and variance predictors

        pitch_z = self.pitch_vae(cond=encoded_texts.transpose(1, 2),
                                 infer=True)
        energy_z = self.energy_vae(cond=encoded_texts.transpose(1, 2),
                                   infer=True)
        duration_z = self.duration_vae(cond=encoded_texts.transpose(1, 2),
                                       infer=True)

        pitch_predictions = self.pitch_vae.decoder(pitch_z,
                                                   nonpadding=None,
                                                   cond=encoded_texts.transpose(1, 2).detach(),
                                                   utt_emb=utterance_embedding).transpose(1, 2)
        energy_predictions = self.energy_vae.decoder(energy_z,
                                                     nonpadding=None,
                                                     cond=encoded_texts.transpose(1, 2).detach(),
                                                     utt_emb=utterance_embedding).transpose(1, 2)
        predicted_durations = self.duration_vae.decoder(duration_z,
                                                        nonpadding=None,
                                                        cond=encoded_texts.transpose(1, 2).detach(),
                                                        utt_emb=utterance_embedding).squeeze(1)

        if gold_durations is not None:
            predicted_durations = gold_durations
        else:
            predicted_durations = make_estimated_durations_usable_for_inference(predicted_durations)
        if gold_pitch is not None:
            pitch_predictions = gold_pitch
        if gold_energy is not None:
            energy_predictions = gold_energy

        for phoneme_index, phoneme_vector in enumerate(text_tensors.squeeze(0)):
            if phoneme_vector[get_feature_to_index_lookup()["voiced"]] == 0:
                # this means the phoneme is unvoiced
                pitch_predictions[0][phoneme_index] = 0.0
            if phoneme_vector[get_feature_to_index_lookup()["silence"]] == 1 and pause_duration_scaling_factor != 1.0:
                predicted_durations[0][phoneme_index] = torch.round(
                    predicted_durations[0][phoneme_index].float() * pause_duration_scaling_factor)
        pitch_predictions = _scale_variance(pitch_predictions, pitch_variance_scale)
        energy_predictions = _scale_variance(energy_predictions, energy_variance_scale)

        embedded_pitch_curve = self.pitch_embed(pitch_predictions.transpose(1, 2)).transpose(1, 2)
        embedded_energy_curve = self.energy_embed(energy_predictions.transpose(1, 2)).transpose(1, 2)
        encoded_texts = encoded_texts + embedded_energy_curve + embedded_pitch_curve
        encoded_texts = self.length_regulator(encoded_texts, predicted_durations,
                                              duration_scaling_factor)  # (B, Lmax, adim)
        predicted_spectrogram_before_postnet = self.decoder(encoded_texts, nonpadding=None,
                                                            utt_emb=utterance_embedding).transpose(1, 2)
        predicted_spectrogram_before_postnet = self.out_proj(predicted_spectrogram_before_postnet).transpose(1, 2)

        # forward flow post-net
        predicted_spectrogram_after_postnet = self.run_post_glow(mel_out=predicted_spectrogram_before_postnet,
                                                                 encoded_texts=encoded_texts,
                                                                 device=device)

        return predicted_spectrogram_before_postnet, predicted_spectrogram_after_postnet, predicted_durations, pitch_predictions, energy_predictions

    @torch.inference_mode()
    def forward(self,
                text,
                durations=None,
                pitch=None,
                energy=None,
                utterance_embedding=None,
                return_duration_pitch_energy=False,
                lang_id=None,
                duration_scaling_factor=1.0,
                pitch_variance_scale=1.0,
                energy_variance_scale=1.0,
                pause_duration_scaling_factor=1.0,
                device=None):
        """
        Generate the sequence of spectrogram frames given the sequence of vectorized phonemes.

        Args:
            text: input sequence of vectorized phonemes
            durations: durations to be used (optional, if not provided, they will be predicted)
            pitch: token-averaged pitch curve to be used (optional, if not provided, it will be predicted)
            energy: token-averaged energy curve to be used (optional, if not provided, it will be predicted)
            return_duration_pitch_energy: whether to return the list of predicted durations for nicer plotting
            utterance_embedding: embedding of speaker information
            lang_id: id to be fed into the embedding layer that contains language information
            duration_scaling_factor: reasonable values are 0.8 < scale < 1.2.
                                     1.0 means no scaling happens, higher values increase durations for the whole
                                     utterance, lower values decrease durations for the whole utterance.
            pitch_variance_scale: reasonable values are 0.6 < scale < 1.4.
                                  1.0 means no scaling happens, higher values increase variance of the pitch curve,
                                  lower values decrease variance of the pitch curve.
            energy_variance_scale: reasonable values are 0.6 < scale < 1.4.
                                   1.0 means no scaling happens, higher values increase variance of the energy curve,
                                   lower values decrease variance of the energy curve.
            pause_duration_scaling_factor: reasonable values are 0.6 < scale < 1.4.
                                   scales the durations of pauses on top of the regular duration scaling

        Returns:
            mel spectrogram

        """
        # setup batch axis
        ilens = torch.tensor([text.shape[0]], dtype=torch.long, device=text.device)
        if durations is not None:
            durations = durations.unsqueeze(0).to(text.device)
        if pitch is not None:
            pitch = pitch.unsqueeze(0).to(text.device)
        if energy is not None:
            energy = energy.unsqueeze(0).to(text.device)
        if lang_id is not None:
            lang_id = lang_id.unsqueeze(0).to(text.device)

        before_outs, \
        after_outs, \
        d_outs, \
        pitch_predictions, \
        energy_predictions = self._forward(text.unsqueeze(0),
                                           ilens,
                                           gold_durations=durations,
                                           gold_pitch=pitch,
                                           gold_energy=energy,
                                           utterance_embedding=utterance_embedding.unsqueeze(0),
                                           lang_ids=lang_id,
                                           duration_scaling_factor=duration_scaling_factor,
                                           pitch_variance_scale=pitch_variance_scale,
                                           energy_variance_scale=energy_variance_scale,
                                           pause_duration_scaling_factor=pause_duration_scaling_factor,
                                           device=device)
        if return_duration_pitch_energy:
            return after_outs[0], d_outs[0], pitch_predictions[0], energy_predictions[0]
        return after_outs[0]

    def _source_mask(self, ilens):
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        return x_masks.unsqueeze(-2)

    def store_inverse_all(self):
        def remove_weight_norm(m):
            try:
                if hasattr(m, 'store_inverse'):
                    m.store_inverse()
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(remove_weight_norm)

    def run_post_glow(self, mel_out, encoded_texts, device):
        x_recon = mel_out.transpose(1, 2)
        g = x_recon
        B, _, T = g.shape
        g = torch.cat([g, encoded_texts.transpose(1, 2)], 1)
        g = self.g_proj(g)
        nonpadding = torch.ones_like(x_recon[:, :1, :])
        z_post = torch.randn(x_recon.shape).to(device) * 0.8
        x_recon, _ = self.post_flow(z_post, nonpadding, g, reverse=True)
        return x_recon.transpose(1, 2)


def _scale_variance(sequence, scale):
    if scale == 1.0:
        return sequence
    average = sequence[0][sequence[0] != 0.0].mean()
    sequence = sequence - average  # center sequence around 0
    sequence = sequence * scale  # scale the variance
    sequence = sequence + average  # move center back to original with changed variance
    for sequence_index in range(len(sequence[0])):
        if sequence[0][sequence_index] < 0.0:
            sequence[0][sequence_index] = 0.0
    return sequence
