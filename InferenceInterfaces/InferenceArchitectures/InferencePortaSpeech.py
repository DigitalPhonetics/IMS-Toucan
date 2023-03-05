import torch
import torch.distributions as dist
from torch.nn import LayerNorm
from torch.nn import Linear
from torch.nn import Sequential

from Layers.Conformer import Conformer
from Layers.DurationPredictor import DurationPredictor
from Layers.LengthRegulator import LengthRegulator
from Layers.VariancePredictor import VariancePredictor
from Preprocessing.articulatory_features import get_feature_to_index_lookup
from TrainingInterfaces.Text_to_Spectrogram.PortaSpeech.Glow import Glow
from Utility.utils import make_non_pad_mask
from Utility.utils import make_pad_mask


class PortaSpeech(torch.nn.Module):

    def __init__(self,
                 # network structure related
                 input_feature_dimensions=62,
                 output_spectrogram_channels=80,
                 attention_dimension=192,
                 attention_heads=4,
                 positionwise_conv_kernel_size=1,
                 use_scaled_positional_encoding=True,
                 # encoder / decoder
                 encoder_layers=6,
                 encoder_units=1536,
                 encoder_normalize_before=True,
                 encoder_concat_after=False,
                 use_macaron_style_in_conformer=True,
                 use_cnn_in_conformer=True,
                 conformer_encoder_kernel_size=7,
                 decoder_layers=6,
                 decoder_units=1536,
                 decoder_concat_after=False,
                 conformer_decoder_kernel_size=31,
                 decoder_normalize_before=True,
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
                 stop_gradient_from_energy_predictor=False,
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
                 # additional features
                 utt_embed_dim=64,
                 lang_embs=8000,
                 weights=None):
        super().__init__()

        # store hyperparameters
        self.idim = input_feature_dimensions
        self.odim = output_spectrogram_channels
        self.adim = attention_dimension
        self.eos = 1
        self.stop_gradient_from_pitch_predictor = stop_gradient_from_pitch_predictor
        self.stop_gradient_from_energy_predictor = stop_gradient_from_energy_predictor
        self.use_scaled_pos_enc = use_scaled_positional_encoding
        self.multilingual_model = lang_embs is not None
        self.multispeaker_model = utt_embed_dim is not None

        # define encoder
        embed = torch.nn.Sequential(torch.nn.Linear(input_feature_dimensions, 100),
                                    torch.nn.Tanh(),
                                    torch.nn.Linear(100, attention_dimension))
        self.encoder = Conformer(idim=input_feature_dimensions,
                                 attention_dim=attention_dimension,
                                 attention_heads=attention_heads,
                                 linear_units=encoder_units,
                                 num_blocks=encoder_layers,
                                 input_layer=embed,
                                 dropout_rate=transformer_enc_dropout_rate,
                                 positional_dropout_rate=transformer_enc_positional_dropout_rate,
                                 attention_dropout_rate=transformer_enc_attn_dropout_rate,
                                 normalize_before=encoder_normalize_before,
                                 concat_after=encoder_concat_after,
                                 positionwise_conv_kernel_size=positionwise_conv_kernel_size,
                                 macaron_style=use_macaron_style_in_conformer,
                                 use_cnn_module=use_cnn_in_conformer,
                                 cnn_module_kernel=conformer_encoder_kernel_size,
                                 zero_triu=False,
                                 utt_embed=None,
                                 lang_embs=lang_embs)

        # define duration predictor
        self.duration_predictor = DurationPredictor(idim=attention_dimension, n_layers=duration_predictor_layers,
                                                    n_chans=duration_predictor_chans,
                                                    kernel_size=duration_predictor_kernel_size,
                                                    dropout_rate=duration_predictor_dropout_rate, )

        # define pitch predictor
        self.pitch_predictor = VariancePredictor(idim=attention_dimension, n_layers=pitch_predictor_layers,
                                                 n_chans=pitch_predictor_chans,
                                                 kernel_size=pitch_predictor_kernel_size,
                                                 dropout_rate=pitch_predictor_dropout)
        self.pitch_embed = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1,
                            out_channels=attention_dimension,
                            kernel_size=pitch_embed_kernel_size,
                            padding=(pitch_embed_kernel_size - 1) // 2),
            torch.nn.Dropout(pitch_embed_dropout))

        # define energy predictor
        self.energy_predictor = VariancePredictor(idim=attention_dimension, n_layers=energy_predictor_layers,
                                                  n_chans=energy_predictor_chans,
                                                  kernel_size=energy_predictor_kernel_size,
                                                  dropout_rate=energy_predictor_dropout)
        self.energy_embed = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1,
                            out_channels=attention_dimension,
                            kernel_size=energy_embed_kernel_size,
                            padding=(energy_embed_kernel_size - 1) // 2),
            torch.nn.Dropout(energy_embed_dropout))

        # define length regulator
        self.length_regulator = LengthRegulator()

        # define decoder
        self.decoder = Conformer(idim=0,
                                 attention_dim=attention_dimension,
                                 attention_heads=attention_heads,
                                 linear_units=decoder_units,
                                 num_blocks=decoder_layers,
                                 input_layer=None,
                                 dropout_rate=transformer_dec_dropout_rate,
                                 positional_dropout_rate=transformer_dec_positional_dropout_rate,
                                 attention_dropout_rate=transformer_dec_attn_dropout_rate,
                                 normalize_before=decoder_normalize_before,
                                 concat_after=decoder_concat_after,
                                 positionwise_conv_kernel_size=positionwise_conv_kernel_size,
                                 macaron_style=use_macaron_style_in_conformer,
                                 use_cnn_module=use_cnn_in_conformer,
                                 cnn_module_kernel=conformer_decoder_kernel_size,
                                 utt_embed=None)

        # define final projection
        self.feat_out = torch.nn.Linear(attention_dimension, output_spectrogram_channels)

        # define speaker embedding integrations
        self.pitch_bottleneck = Linear(utt_embed_dim, utt_embed_dim // 2)
        self.energy_bottleneck = Linear(utt_embed_dim, utt_embed_dim // 2)
        self.duration_bottleneck = Linear(utt_embed_dim, utt_embed_dim // 2)
        self.pitch_embedding_projection = Sequential(Linear(attention_dimension + utt_embed_dim // 2, attention_dimension),
                                                     LayerNorm(attention_dimension))
        self.energy_embedding_projection = Sequential(Linear(attention_dimension + utt_embed_dim // 2,
                                                             attention_dimension),
                                                      LayerNorm(attention_dimension))
        self.duration_embedding_projection = Sequential(Linear(attention_dimension + utt_embed_dim // 2,
                                                               attention_dimension),
                                                        LayerNorm(attention_dimension))
        self.decoder_in_embedding_projection = Sequential(Linear(attention_dimension + utt_embed_dim, attention_dimension),
                                                          LayerNorm(attention_dimension))
        self.decoder_out_embedding_projection = Sequential(Linear(output_spectrogram_channels + utt_embed_dim,
                                                                  output_spectrogram_channels),
                                                           LayerNorm(output_spectrogram_channels))

        # post net is realized as a flow
        gin_channels = attention_dimension
        self.post_flow = Glow(
            output_spectrogram_channels,
            192,  # post_glow_hidden  (original 192 in paper)
            3,  # post_glow_kernel_size
            1,
            16,  # post_glow_n_blocks
            3,  # post_glow_n_block_layers
            n_split=4,
            n_sqz=2,
            gin_channels=gin_channels,
            share_cond_layers=False,  # post_share_cond_layers
            share_wn_layers=4,  # share_wn_layers
            sigmoid_scale=False  # sigmoid_scale
        )
        self.prior_dist = dist.Normal(0, 1)

        self.g_proj = torch.nn.Conv1d(output_spectrogram_channels + attention_dimension, gin_channels, 5, padding=2)

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

        if utterance_embedding is not None:

            encoded_texts_for_pitch = _integrate_with_utt_embed(hs=encoded_texts,
                                                                utt_embeddings=self.pitch_bottleneck(utterance_embedding),
                                                                projection=self.pitch_embedding_projection)
            encoded_texts_for_energy = _integrate_with_utt_embed(hs=encoded_texts,
                                                                 utt_embeddings=self.energy_bottleneck(utterance_embedding),
                                                                 projection=self.energy_embedding_projection)
            encoded_texts_for_duration = _integrate_with_utt_embed(hs=encoded_texts,
                                                                   utt_embeddings=self.duration_bottleneck(utterance_embedding),
                                                                   projection=self.duration_embedding_projection)
        else:
            encoded_texts_for_pitch = encoded_texts
            encoded_texts_for_energy = encoded_texts
            encoded_texts_for_duration = encoded_texts

        # forward duration predictor and variance predictors
        d_masks = make_pad_mask(text_lens, device=text_lens.device)

        if gold_durations is not None:
            predicted_durations = gold_durations
        else:
            predicted_durations = self.duration_predictor.inference(encoded_texts_for_duration, d_masks)
        if gold_pitch is not None:
            pitch_predictions = gold_pitch
        else:
            pitch_predictions = self.pitch_predictor(encoded_texts_for_pitch, d_masks.unsqueeze(-1))
        if gold_energy is not None:
            energy_predictions = gold_energy
        else:
            energy_predictions = self.energy_predictor(encoded_texts_for_energy, d_masks.unsqueeze(-1))

        for phoneme_index, phoneme_vector in enumerate(text_tensors.squeeze(0)):
            if phoneme_vector[get_feature_to_index_lookup()["questionmark"]] == 1:
                if phoneme_index - 4 >= 0:
                    pitch_predictions[0][phoneme_index - 1] += .3
                    pitch_predictions[0][phoneme_index - 2] += .3
                    pitch_predictions[0][phoneme_index - 3] += .2
                    pitch_predictions[0][phoneme_index - 4] += .1
            if phoneme_vector[get_feature_to_index_lookup()["exclamationmark"]] == 1:
                if phoneme_index - 6 >= 0:
                    pitch_predictions[0][phoneme_index - 1] += .3
                    pitch_predictions[0][phoneme_index - 2] += .3
                    pitch_predictions[0][phoneme_index - 3] += .2
                    pitch_predictions[0][phoneme_index - 4] += .2
                    pitch_predictions[0][phoneme_index - 5] += .1
                    pitch_predictions[0][phoneme_index - 6] += .1
            if phoneme_vector[get_feature_to_index_lookup()["voiced"]] == 0:
                # this means the phoneme is unvoiced and should therefore not have a pitch value (undefined, but we overload this with 0)
                pitch_predictions[0][phoneme_index] = 0.0
            if phoneme_vector[get_feature_to_index_lookup()["word-boundary"]] == 1:
                predicted_durations[0][phoneme_index] = 0
            if phoneme_vector[get_feature_to_index_lookup()["silence"]] == 1 and pause_duration_scaling_factor != 1.0:
                predicted_durations[0][phoneme_index] = torch.round(predicted_durations[0][phoneme_index].float() * pause_duration_scaling_factor).long()
        if duration_scaling_factor != 1.0:
            assert duration_scaling_factor > 0
            predicted_durations = torch.round(predicted_durations.float() * duration_scaling_factor).long()
        pitch_predictions = _scale_variance(pitch_predictions, pitch_variance_scale)
        energy_predictions = _scale_variance(energy_predictions, energy_variance_scale)

        embedded_pitch_curve = self.pitch_embed(pitch_predictions.transpose(1, 2)).transpose(1, 2)
        embedded_energy_curve = self.energy_embed(energy_predictions.transpose(1, 2)).transpose(1, 2)
        encoded_texts = encoded_texts + embedded_energy_curve + embedded_pitch_curve
        encoded_texts = self.length_regulator(encoded_texts, predicted_durations)

        if utterance_embedding is not None:
            encoded_texts = _integrate_with_utt_embed(hs=encoded_texts,
                                                      utt_embeddings=utterance_embedding,
                                                      projection=self.decoder_in_embedding_projection)

        decoded_speech, _ = self.decoder(encoded_texts, None, utterance_embedding)
        predicted_spectrogram_before_postnet = self.feat_out(decoded_speech).view(decoded_speech.size(0), -1, self.odim)

        # forward flow post-net
        if utterance_embedding is not None:
            before_enriched = _integrate_with_utt_embed(hs=predicted_spectrogram_before_postnet,
                                                        utt_embeddings=utterance_embedding,
                                                        projection=self.decoder_out_embedding_projection)
        else:
            before_enriched = predicted_spectrogram_before_postnet
        predicted_spectrogram_after_postnet = self.run_post_glow(mel_out=before_enriched,
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


def _integrate_with_utt_embed(hs, utt_embeddings, projection):
    # concat hidden states with spk embeds and then apply projection
    embeddings_expanded = torch.nn.functional.normalize(utt_embeddings).unsqueeze(1).expand(-1, hs.size(1), -1)
    hs = projection(torch.cat([hs, embeddings_expanded], dim=-1))
    return hs


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
