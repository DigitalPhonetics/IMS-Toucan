import random
from abc import ABC

import torch
import torch.distributions as dist
from torch.nn import LayerNorm
from torch.nn import Linear
from torch.nn import Sequential
from torch.nn import Tanh

from Layers.Conformer import Conformer
from Layers.LengthRegulator import LengthRegulator
from Preprocessing.articulatory_features import get_feature_to_index_lookup
from TrainingInterfaces.Text_to_Spectrogram.PortaSpeech.Glow import Glow
from TrainingInterfaces.Text_to_Spectrogram.ToucanTTS.ToucanTTSLossWGAN import ToucanTTSLoss
from TrainingInterfaces.Text_to_Spectrogram.ToucanTTS.VarianceDiscriminator import VarianceDiscriminator
from TrainingInterfaces.Text_to_Spectrogram.ToucanTTS.VariationalVariancePredictor import VariationalVariancePredictor
from Utility.utils import initialize
from Utility.utils import make_estimated_durations_usable_for_inference
from Utility.utils import make_non_pad_mask


class ToucanTTS(torch.nn.Module, ABC):
    """
    ToucanTTS module, which is basically just a FastSpeech 2 module,
    but with variance predictors that make more sense and the PostNet
    is a normalizing flow, like in PortaSpeech. Furthermore, the pitch
    and energy values are averaged per-phone, as in FastPitch. The
    encoder and decoder are Conformers. Input features are configurations
    of the articulatory tract rather than phoneme identities. Speaker
    embedding conditioning is derived from GST and Adaspeech 4. There is
    a large focus on multilinguality through numerous designs.
    """

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
                 init_type="xavier_uniform",
                 init_enc_alpha=1.0,
                 init_dec_alpha=1.0,
                 # additional features
                 utt_embed_dim=64,
                 detach_postflow=False,
                 lang_embs=8000,
                 window_size=5):
        super().__init__()

        # store hyperparameters
        self.idim = input_feature_dimensions
        self.odim = output_spectrogram_channels
        self.adim = attention_dimension
        self.eos = 1
        self.window_size = window_size
        self.detach_postflow = detach_postflow
        self.stop_gradient_from_pitch_predictor = stop_gradient_from_pitch_predictor
        self.stop_gradient_from_energy_predictor = stop_gradient_from_energy_predictor
        self.use_scaled_pos_enc = use_scaled_positional_encoding
        self.multilingual_model = lang_embs is not None
        self.multispeaker_model = utt_embed_dim is not None

        # define encoder
        embed = Sequential(Linear(input_feature_dimensions, 100),
                           Tanh(),
                           Linear(100, attention_dimension))
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
        self.duration_predictor = VariationalVariancePredictor(out_channels=1,  # 1 dimensional output sequence
                                                               hidden_size=duration_predictor_chans,  # size of embedding space
                                                               c_latent=duration_predictor_chans,  # latent space inbetween encoder and decoder
                                                               kernel_size=duration_predictor_kernel_size,
                                                               n_layers=duration_predictor_layers,
                                                               c_cond=attention_dimension,  # condition to guide the sampling
                                                               strides=[1],
                                                               # norm_type="cln" if utt_embed_dim is not None else "ln",
                                                               norm_type="ln",
                                                               spk_emb_size=utt_embed_dim)

        # define pitch predictor
        self.pitch_predictor = VariationalVariancePredictor(out_channels=1,
                                                            hidden_size=pitch_predictor_chans,
                                                            c_latent=pitch_predictor_chans,
                                                            kernel_size=pitch_predictor_kernel_size,
                                                            n_layers=pitch_predictor_layers,
                                                            c_cond=attention_dimension,
                                                            strides=[1],
                                                            # norm_type="cln" if utt_embed_dim is not None else "ln",
                                                            norm_type="ln", spk_emb_size=utt_embed_dim)

        self.pitch_embed = Sequential(
            torch.nn.Conv1d(in_channels=1,
                            out_channels=attention_dimension,
                            kernel_size=pitch_embed_kernel_size,
                            padding=(pitch_embed_kernel_size - 1) // 2),
            torch.nn.Dropout(pitch_embed_dropout))

        # define energy predictor
        self.energy_predictor = VariationalVariancePredictor(out_channels=1,
                                                             hidden_size=energy_predictor_chans,
                                                             c_latent=energy_predictor_chans,
                                                             kernel_size=energy_predictor_kernel_size,
                                                             n_layers=energy_predictor_layers,
                                                             c_cond=attention_dimension,
                                                             strides=[1],
                                                             # norm_type="cln" if utt_embed_dim is not None else "ln",
                                                             norm_type="ln",
                                                             spk_emb_size=utt_embed_dim)
        self.energy_embed = Sequential(
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
        self.feat_out = Linear(attention_dimension, output_spectrogram_channels)

        # define speaker embedding integrations
        self.decoder_in_embedding_projection = Sequential(Linear(attention_dimension + utt_embed_dim, attention_dimension), LayerNorm(attention_dimension))
        self.variance_embedding_projection = Sequential(Linear(attention_dimension + utt_embed_dim, attention_dimension), LayerNorm(attention_dimension))
        self.decoder_out_embedding_projection = Sequential(Linear(output_spectrogram_channels + utt_embed_dim, output_spectrogram_channels), LayerNorm(output_spectrogram_channels))

        # post net is realized as a flow
        gin_channels = attention_dimension
        self.post_flow = Glow(
            in_channels=output_spectrogram_channels,
            hidden_channels=192,  # post_glow_hidden  (original 192 in paper)
            kernel_size=3,  # post_glow_kernel_size
            dilation_rate=1,
            n_blocks=16,  # post_glow_n_blocks (original 12 in paper)
            n_layers=3,  # post_glow_n_block_layers (original 3 in paper)
            n_split=4,
            n_sqz=2,
            gin_channels=gin_channels,
            share_cond_layers=False,  # post_share_cond_layers
            share_wn_layers=4,  # share_wn_layers
            sigmoid_scale=False  # sigmoid_scale
        )
        self.prior_dist = dist.Normal(0, 1)

        self.g_proj = torch.nn.Conv1d(output_spectrogram_channels + attention_dimension, gin_channels, 5, padding=2)

        # initialize parameters
        self._reset_parameters(init_type=init_type, init_enc_alpha=init_enc_alpha, init_dec_alpha=init_dec_alpha)
        if lang_embs is not None:
            torch.nn.init.normal_(self.encoder.language_embedding.weight, mean=0, std=attention_dimension ** -0.5)

        self.duration_discriminator = VarianceDiscriminator(data_dim=[1, window_size, attention_dimension + 1])
        self.pitch_discriminator = VarianceDiscriminator(data_dim=[1, window_size, attention_dimension + 1])
        self.energy_discriminator = VarianceDiscriminator(data_dim=[1, window_size, attention_dimension + 1])

        # define criterion
        self.criterion = ToucanTTSLoss()

    def forward(self,
                text_tensors,
                text_lengths,
                gold_speech,
                speech_lengths,
                gold_durations,
                gold_pitch,
                gold_energy,
                utterance_embedding,
                return_mels=False,
                lang_ids=None,
                run_glow=True
                ):
        """
        Calculate forward propagation.

        Args:
            return_mels: whether to return the predicted spectrogram
            text_tensors (LongTensor): Batch of padded text vectors (B, Tmax).
            text_lengths (LongTensor): Batch of lengths of each input (B,).
            gold_speech (Tensor): Batch of padded target features (B, Lmax, odim).
            speech_lengths (LongTensor): Batch of the lengths of each target (B,).
            gold_durations (LongTensor): Batch of padded durations (B, Tmax + 1).
            gold_pitch (Tensor): Batch of padded token-averaged pitch (B, Tmax + 1, 1).
            gold_energy (Tensor): Batch of padded token-averaged energy (B, Tmax + 1, 1).

        Returns:
            Tensor: Loss scalar value.
            Dict: Statistics to be monitored.
            Tensor: Weight value.
        """
        # Texts include EOS token from the teacher model already in this version

        # forward propagation
        before_outs, \
            after_outs, \
            d_outs, \
            p_outs, \
            e_outs, \
            glow_loss = self._forward(text_tensors,
                                      text_lengths,
                                      gold_speech,
                                      speech_lengths,
                                      gold_durations,
                                      gold_pitch,
                                      gold_energy,
                                      utterance_embedding=utterance_embedding,
                                      is_inference=False,
                                      lang_ids=lang_ids,
                                      run_glow=run_glow)
        d_outs = d_outs.squeeze(1)
        p_outs = p_outs.transpose(1, 2)
        e_outs = e_outs.transpose(1, 2)

        # calculate loss
        l1_loss, duration_loss, pitch_loss, energy_loss = self.criterion(after_outs=None,
                                                                         # if a regular postnet is used, the post-postnet outs have to go here. The flow has its own loss though, so we hard-code this to None
                                                                         before_outs=before_outs,
                                                                         d_outs=d_outs,
                                                                         p_outs=p_outs,
                                                                         e_outs=e_outs,
                                                                         ys=gold_speech,
                                                                         ds=gold_durations,
                                                                         ps=gold_pitch,
                                                                         es=gold_energy,
                                                                         ilens=text_lengths,
                                                                         olens=speech_lengths)

        kl_loss = torch.tensor([0.0], device=l1_loss.device)  # for future extension / legacy compatibility
        if return_mels:
            if after_outs is None:
                after_outs = before_outs
            return l1_loss, duration_loss, pitch_loss, energy_loss, glow_loss, kl_loss, after_outs
        return l1_loss, duration_loss, pitch_loss, energy_loss, glow_loss, kl_loss

    def _forward(self,
                 text_tensors,
                 text_lens,
                 gold_speech=None,
                 speech_lens=None,
                 gold_durations=None,
                 gold_pitch=None,
                 gold_energy=None,
                 is_inference=False,
                 alpha=1.0,
                 utterance_embedding=None,
                 lang_ids=None,
                 run_glow=True):

        if not self.multilingual_model:
            lang_ids = None

        if not self.multispeaker_model:
            utterance_embedding = None

        # forward text encoder
        text_masks = self._source_mask(text_lens)

        encoded_texts, _ = self.encoder(text_tensors,
                                        text_masks,
                                        utterance_embedding=utterance_embedding,
                                        lang_ids=lang_ids)

        text_nonpadding_mask = make_non_pad_mask(text_lens, device=text_lens.device)

        if utterance_embedding is not None:
            encoded_texts_with_speaker_info = _integrate_with_utt_embed(hs=encoded_texts,
                                                                        utt_embeddings=utterance_embedding,
                                                                        projection=self.variance_embedding_projection)
        else:
            encoded_texts_with_speaker_info = encoded_texts

        if is_inference:
            pitch_predictions = self.pitch_predictor(cond=encoded_texts_with_speaker_info.transpose(1, 2), utt_emb=utterance_embedding)
            embedded_pitch_curve = self.pitch_embed(pitch_predictions).transpose(1, 2)
            encoded_texts_enriched_with_pitch = encoded_texts_with_speaker_info + embedded_pitch_curve

            energy_predictions = self.energy_predictor(cond=encoded_texts_enriched_with_pitch.transpose(1, 2), utt_emb=utterance_embedding)
            embedded_energy_curve = self.energy_embed(energy_predictions).transpose(1, 2)
            enriched_encoded_texts = encoded_texts + embedded_energy_curve + embedded_pitch_curve

            predicted_durations = self.duration_predictor(cond=enriched_encoded_texts.transpose(1, 2), utt_emb=utterance_embedding)
            upsampled_enriched_encoded_texts = self.length_regulator(enriched_encoded_texts, make_estimated_durations_usable_for_inference(predicted_durations).squeeze(0), alpha)

        else:
            # training with teacher forcing
            pitch_predictions = self.pitch_predictor(nonpadding=text_nonpadding_mask.unsqueeze(1), cond=encoded_texts_with_speaker_info.transpose(1, 2), utt_emb=utterance_embedding)
            embedded_pitch_curve = self.pitch_embed(gold_pitch.transpose(1, 2)).transpose(1, 2)
            encoded_texts_enriched_with_pitch = encoded_texts_with_speaker_info + embedded_pitch_curve

            energy_predictions = self.energy_predictor(nonpadding=text_nonpadding_mask.unsqueeze(1), cond=encoded_texts_enriched_with_pitch.transpose(1, 2), utt_emb=utterance_embedding)
            embedded_energy_curve = self.energy_embed(gold_energy.transpose(1, 2)).transpose(1, 2)
            enriched_encoded_texts = encoded_texts + embedded_energy_curve + embedded_pitch_curve

            predicted_durations = self.duration_predictor(nonpadding=text_nonpadding_mask.unsqueeze(1), cond=enriched_encoded_texts.transpose(1, 2), utt_emb=utterance_embedding)
            upsampled_enriched_encoded_texts = self.length_regulator(enriched_encoded_texts, gold_durations, alpha)

        # forward the decoder
        if not is_inference:
            speech_nonpadding_mask = make_non_pad_mask(lengths=speech_lens,
                                                       device=speech_lens.device).unsqueeze(1).transpose(1, 2)
        else:
            speech_nonpadding_mask = None

        if speech_lens is not None and not is_inference:
            decoder_masks = self._source_mask(speech_lens)
        else:
            decoder_masks = None

        if utterance_embedding is not None:
            upsampled_enriched_encoded_texts = _integrate_with_utt_embed(hs=upsampled_enriched_encoded_texts,
                                                                         utt_embeddings=utterance_embedding,
                                                                         projection=self.decoder_in_embedding_projection)

        decoded_speech, _ = self.decoder(upsampled_enriched_encoded_texts, decoder_masks, utterance_embedding)
        predicted_spectrogram_before_postnet = self.feat_out(decoded_speech).view(decoded_speech.size(0), -1, self.odim)
        if self.detach_postflow:
            predicted_spectrogram_before_postnet = predicted_spectrogram_before_postnet.detach()

        predicted_spectrogram_after_postnet = None

        # forward flow post-net
        glow_loss = None
        if run_glow:
            if utterance_embedding is not None:
                before_enriched = _integrate_with_utt_embed(hs=predicted_spectrogram_before_postnet,
                                                            utt_embeddings=utterance_embedding,
                                                            projection=self.decoder_out_embedding_projection)
            else:
                before_enriched = predicted_spectrogram_before_postnet

            if is_inference:
                predicted_spectrogram_after_postnet = self.run_post_glow(tgt_mels=None,
                                                                         infer=is_inference,
                                                                         mel_out=before_enriched,
                                                                         encoded_texts=upsampled_enriched_encoded_texts,
                                                                         tgt_nonpadding=None)


            else:
                glow_loss = self.run_post_glow(tgt_mels=gold_speech,
                                               infer=is_inference,
                                               mel_out=before_enriched,
                                               encoded_texts=upsampled_enriched_encoded_texts,
                                               tgt_nonpadding=speech_nonpadding_mask.transpose(1, 2))
        if is_inference:
            return predicted_spectrogram_before_postnet, \
                predicted_spectrogram_after_postnet, \
                predicted_durations, \
                pitch_predictions, \
                energy_predictions
        else:
            return predicted_spectrogram_before_postnet, \
                predicted_spectrogram_after_postnet, \
                predicted_durations, \
                pitch_predictions, \
                energy_predictions, \
                glow_loss

    def inference(self,
                  text,
                  speech=None,
                  durations=None,
                  pitch=None,
                  energy=None,
                  alpha=1.0,
                  utterance_embedding=None,
                  return_duration_pitch_energy=False,
                  lang_id=None,
                  run_postflow=True):
        """
        Generate the sequence of features given the sequences of characters.

        Args:
            text (LongTensor): Input sequence of characters (T,).
            speech (Tensor, optional): Feature sequence to extract style (N, idim).
            durations (LongTensor, optional): Groundtruth of duration (T + 1,).
            pitch (Tensor, optional): Groundtruth of token-averaged pitch (T + 1, 1).
            energy (Tensor, optional): Groundtruth of token-averaged energy (T + 1, 1).
            alpha (float, optional): Alpha to control the speed.
            return_duration_pitch_energy: whether to return the list of predicted durations for nicer plotting

        Returns:
            Tensor: Output sequence of features (L, odim).

        """
        self.eval()
        x, y = text, speech
        d, p, e = durations, pitch, energy

        # setup batch axis
        ilens = torch.tensor([x.shape[0]], dtype=torch.long, device=x.device)
        xs, ys = x.unsqueeze(0), None
        if y is not None:
            ys = y.unsqueeze(0)
        if lang_id is not None:
            lang_id = lang_id.unsqueeze(0)

        before_outs, \
            after_outs, \
            d_outs, \
            pitch_predictions, \
            energy_predictions = self._forward(xs,
                                               ilens,
                                               ys,
                                               is_inference=True,
                                               alpha=alpha,
                                               utterance_embedding=utterance_embedding.unsqueeze(0),
                                               lang_ids=lang_id,
                                               run_glow=run_postflow)  # (1, L, odim)
        for phoneme_index, phoneme_vector in enumerate(xs.squeeze()):
            if phoneme_vector[get_feature_to_index_lookup()["voiced"]] == 0:
                pitch_predictions[0][phoneme_index] = 0.0
        self.train()
        if after_outs is None:
            after_outs = before_outs
        if return_duration_pitch_energy:
            return (before_outs[0], after_outs[0]), d_outs[0], pitch_predictions[0], energy_predictions[0]
        return after_outs[0]

    def run_post_glow(self, tgt_mels, infer, mel_out, encoded_texts, tgt_nonpadding):
        x_recon = mel_out.transpose(1, 2)
        g = x_recon
        B, _, T = g.shape
        g = torch.cat([g, encoded_texts.transpose(1, 2)], 1)
        g = self.g_proj(g)
        prior_dist = self.prior_dist
        if not infer:
            y_lengths = tgt_nonpadding.sum(-1)
            tgt_mels = tgt_mels.transpose(1, 2)
            z_postflow, ldj = self.post_flow(tgt_mels, tgt_nonpadding, g=g)
            ldj = ldj / y_lengths / 80
            try:
                postflow_loss = -prior_dist.log_prob(z_postflow).mean() - ldj.mean()
            except ValueError:
                print("log probability of plostflow could not be calculated for this step")
                postflow_loss = None
            return postflow_loss
        else:
            nonpadding = torch.ones_like(x_recon[:, :1, :])
            z_post = torch.randn(x_recon.shape).to(g.device) * 0.8
            x_recon, _ = self.post_flow(z_post, nonpadding, g, reverse=True)
            return x_recon.transpose(1, 2)

    def _source_mask(self, ilens):
        # Make masks for self-attention.
        x_masks = make_non_pad_mask(ilens, device=ilens.device)
        return x_masks.unsqueeze(-2)

    def _reset_parameters(self, init_type, init_enc_alpha, init_dec_alpha):
        # initialize parameters
        if init_type != "pytorch":
            initialize(self, init_type)

    def initialize_solver(self, batch_size):
        self.duration_discriminator.initialize_solver(batch_size=batch_size)
        self.pitch_discriminator.initialize_solver(batch_size=batch_size)
        self.energy_discriminator.initialize_solver(batch_size=batch_size)

    def calculate_discriminator_losses(self,
                                       text_tensors,
                                       text_lens,
                                       gold_durations,
                                       gold_pitch,
                                       gold_energy,
                                       utterance_embedding,
                                       lang_ids=None,
                                       ):
        """
        Basically a forward pass that only returns the discriminator loss, because it requires more steps than the rest.
        """
        if not self.multilingual_model:
            lang_ids = None

        if not self.multispeaker_model:
            utterance_embedding = None

        # forward text encoder
        text_masks = self._source_mask(text_lens)

        encoded_texts, _ = self.encoder(text_tensors,
                                        text_masks,
                                        utterance_embedding=utterance_embedding,
                                        lang_ids=lang_ids)

        text_nonpadding_mask = make_non_pad_mask(text_lens, device=text_lens.device)

        if utterance_embedding is not None:
            encoded_texts_with_speaker_info = _integrate_with_utt_embed(hs=encoded_texts,
                                                                        utt_embeddings=utterance_embedding,
                                                                        projection=self.variance_embedding_projection)
        else:
            encoded_texts_with_speaker_info = encoded_texts

        # training with teacher forcing
        pitch_predictions = self.pitch_predictor(nonpadding=text_nonpadding_mask.unsqueeze(1), cond=encoded_texts_with_speaker_info.transpose(1, 2), utt_emb=utterance_embedding)
        embedded_pitch_curve = self.pitch_embed(gold_pitch.transpose(1, 2)).transpose(1, 2)
        encoded_texts_enriched_with_pitch = encoded_texts_with_speaker_info + embedded_pitch_curve

        energy_predictions = self.energy_predictor(nonpadding=text_nonpadding_mask.unsqueeze(1), cond=encoded_texts_enriched_with_pitch.transpose(1, 2), utt_emb=utterance_embedding)
        embedded_energy_curve = self.energy_embed(gold_energy.transpose(1, 2)).transpose(1, 2)
        enriched_encoded_texts = encoded_texts + embedded_energy_curve + embedded_pitch_curve

        predicted_durations = self.duration_predictor(nonpadding=text_nonpadding_mask.unsqueeze(1), cond=enriched_encoded_texts.transpose(1, 2), utt_emb=utterance_embedding)

        # calculate discriminator losses
        # the windowing function is inefficient because it loops over the text lens. We could merge these three calls into one in the future.
        pitch_f_window, pitch_r_window, pitch_cond_window = self.get_random_window(pitch_predictions.transpose(1, 2), gold_pitch, encoded_texts.detach(), text_lens)
        energy_f_window, energy_r_window, energy_cond_window = self.get_random_window(energy_predictions.transpose(1, 2), gold_energy, encoded_texts_enriched_with_pitch.detach(), text_lens)
        duration_f_window, duration_r_window, duration_cond_window = self.get_random_window(predicted_durations.transpose(1, 2), gold_durations.unsqueeze(2), enriched_encoded_texts.detach(), text_lens)

        # [Batch, Sequence, Hidden]
        pitch_critic_loss, pitch_generator_loss = self.pitch_discriminator.train_step(pitch_f_window.unsqueeze(1), pitch_r_window.unsqueeze(1), pitch_cond_window.unsqueeze(1))
        energy_critic_loss, energy_generator_loss = self.energy_discriminator.train_step(energy_f_window.unsqueeze(1), energy_r_window.unsqueeze(1), energy_cond_window.unsqueeze(1))
        duration_critic_loss, duration_generator_loss = self.duration_discriminator.train_step(duration_f_window.unsqueeze(1), duration_r_window.unsqueeze(1), duration_cond_window.unsqueeze(1))

        return pitch_critic_loss, energy_critic_loss, duration_critic_loss, pitch_generator_loss, energy_generator_loss, duration_generator_loss

    def get_random_window(self, generated_sequences, real_sequences, condition_sequences, text_lens):
        """
        This will return a randomized but consistent window of each that can be passed to the discriminator
        Suboptimal runtime because of a loop, should not be too bad, but a fix would be nice.
        """
        generated_windows = list()
        real_windows = list()
        condition_windows = list()

        for end_index, generated, real, condition in zip(text_lens.squeeze(), generated_sequences, real_sequences, condition_sequences):
            max_start = end_index - self.window_size
            if max_start > 0:
                start = random.randint(0, max_start)
            else:
                start = 0
            generated_windows.append(generated[start:start + self.window_size].unsqueeze(0))
            real_windows.append(real[start:start + self.window_size].unsqueeze(0))
            condition_windows.append(condition[start:start + self.window_size].unsqueeze(0))
        return torch.cat(generated_windows, dim=0), torch.cat(real_windows, dim=0), torch.cat(condition_windows, dim=0)


def _integrate_with_utt_embed(hs, utt_embeddings, projection):
    # concat hidden states with spk embeds and then apply projection
    embeddings_expanded = torch.nn.functional.normalize(utt_embeddings).unsqueeze(1).expand(-1, hs.size(1), -1)
    hs = projection(torch.cat([hs, embeddings_expanded], dim=-1))
    return hs


if __name__ == '__main__':
    print(""" TESTING INFERENCE """)
    dummy_text_batch = torch.randn([12, 62])  # [Sequence Length, Features per Phone]
    dummy_utterance_embed = torch.randn([64])  # [Dimensions of Speaker Embedding]
    dummy_language_id = torch.LongTensor([2])
    print(ToucanTTS().inference(dummy_text_batch,
                                utterance_embedding=dummy_utterance_embed,
                                lang_id=dummy_language_id).shape)

    print(""" TESTING TRAINING """)

    dummy_text_batch = torch.randn([2, 3, 62])  # [Batch, Sequence Length, Features per Phone]
    dummy_text_lens = torch.LongTensor([2, 3])

    dummy_speech_batch = torch.randn([2, 30, 80])  # [Batch, Sequence Length, Spectrogram Buckets]
    dummy_speech_lens = torch.LongTensor([10, 30])

    dummy_durations = torch.LongTensor([[10, 0, 0], [10, 15, 5]])
    dummy_pitch = torch.Tensor([[[1.0], [0.], [0.]], [[1.1], [1.2], [0.8]]])
    dummy_energy = torch.tensor([[[1.0], [0.], [0.]], [[1.1], [1.2], [0.8]]])

    dummy_utterance_embed = torch.randn([2, 64])  # [Batch, Dimensions of Speaker Embedding]
    dummy_language_id = torch.LongTensor([5, 3]).unsqueeze(1)

    model = ToucanTTS(window_size=2)
    model.initialize_solver(2)
    l1_loss, duration_loss, pitch_loss, energy_loss, glow_loss, kl_loss = model(dummy_text_batch,
                                                                                dummy_text_lens,
                                                                                dummy_speech_batch,
                                                                                dummy_speech_lens,
                                                                                dummy_durations,
                                                                                dummy_pitch,
                                                                                dummy_energy,
                                                                                utterance_embedding=dummy_utterance_embed,
                                                                                lang_ids=dummy_language_id)

    loss = l1_loss + glow_loss + duration_loss + energy_loss + pitch_loss
    print(loss)
    loss.backward()

    print(model.calculate_discriminator_losses(dummy_text_batch,
                                               dummy_text_lens,
                                               dummy_durations,
                                               dummy_pitch,
                                               dummy_energy,
                                               utterance_embedding=dummy_utterance_embed,
                                               lang_ids=dummy_language_id))

    print(sum(p.numel() for p in ToucanTTS().parameters() if p.requires_grad))
