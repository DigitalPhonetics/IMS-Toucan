from abc import ABC

import torch
import torch.distributions as dist

from Layers.Conformer import Conformer
from Layers.LengthRegulator import LengthRegulator
from Preprocessing.articulatory_features import get_feature_to_index_lookup
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2Loss import FastSpeech2Loss
from TrainingInterfaces.Text_to_Spectrogram.PortaSpeech.FVAE import FVAE
from TrainingInterfaces.Text_to_Spectrogram.PortaSpeech.Glow import Glow
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
                 duration_predictor_layers=4,
                 duration_predictor_chans=64,
                 duration_predictor_kernel_size=3,
                 # energy predictor
                 energy_predictor_layers=3,
                 energy_predictor_chans=64,
                 energy_predictor_kernel_size=3,
                 energy_predictor_dropout=0.5,
                 energy_embed_kernel_size=1,
                 energy_embed_dropout=0.0,
                 stop_gradient_from_energy_predictor=False,
                 # pitch predictor
                 pitch_predictor_layers=5,
                 pitch_predictor_chans=64,
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
                 use_masking=False,
                 use_weighted_masking=True,
                 # additional features
                 utt_embed_dim=256,
                 lang_embs=8000):
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
                                 utt_embed=utt_embed_dim,
                                 lang_embs=lang_embs)

        self.embedding_prenet_for_variance_predictors = torch.nn.Sequential(
            torch.nn.Linear(utt_embed_dim + attention_dimension, utt_embed_dim + attention_dimension),
            torch.nn.Tanh(),
            torch.nn.Linear(utt_embed_dim + attention_dimension, utt_embed_dim + attention_dimension),
            torch.nn.Tanh(),
            torch.nn.Linear(utt_embed_dim + attention_dimension, utt_embed_dim)
        )

        # define duration predictor
        self.duration_vae = FVAE(c_in=1,  # 1 dimensional random variable based sequence
                                 c_out=1,  # 1 dimensional output sequence
                                 hidden_size=duration_predictor_chans,  # size of embedding space
                                 c_latent=duration_predictor_chans // 12,  # latent space inbetween encoder and decoder
                                 kernel_size=duration_predictor_kernel_size,
                                 enc_n_layers=duration_predictor_layers * 2,
                                 dec_n_layers=duration_predictor_layers,
                                 c_cond=attention_dimension,  # condition to guide the sampling
                                 strides=[1],
                                 use_prior_flow=False,
                                 norm_type="cln" if utt_embed_dim is not None else "ln",
                                 spk_emb_size=utt_embed_dim)

        # define pitch predictor
        self.pitch_vae = FVAE(c_in=1,
                              c_out=1,
                              hidden_size=pitch_predictor_chans,
                              c_latent=pitch_predictor_chans // 12,
                              kernel_size=pitch_predictor_kernel_size,
                              enc_n_layers=pitch_predictor_layers * 2,
                              dec_n_layers=pitch_predictor_layers,
                              c_cond=attention_dimension,
                              strides=[1],
                              use_prior_flow=False,
                              norm_type="cln" if utt_embed_dim is not None else "ln",
                              spk_emb_size=utt_embed_dim)
        self.pitch_embed = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1,
                            out_channels=attention_dimension,
                            kernel_size=pitch_embed_kernel_size,
                            padding=(pitch_embed_kernel_size - 1) // 2),
            torch.nn.Dropout(pitch_embed_dropout))

        # define energy predictor
        self.energy_vae = FVAE(c_in=1,
                               c_out=1,
                               hidden_size=energy_predictor_chans,
                               c_latent=energy_predictor_chans // 12,
                               kernel_size=energy_predictor_kernel_size,
                               enc_n_layers=energy_predictor_layers * 2,
                               dec_n_layers=energy_predictor_layers,
                               c_cond=attention_dimension,
                               strides=[1],
                               use_prior_flow=False,
                               norm_type="cln" if utt_embed_dim is not None else "ln",
                               spk_emb_size=utt_embed_dim)
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
                                 utt_embed=utt_embed_dim)

        # define final projection
        self.feat_out = torch.nn.Linear(attention_dimension, output_spectrogram_channels)

        # post net is realized as a flow
        gin_channels = attention_dimension
        self.post_flow = Glow(
            output_spectrogram_channels,
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

        self.g_proj = torch.nn.Conv1d(output_spectrogram_channels + attention_dimension, gin_channels, 5, padding=2)

        # initialize parameters
        self._reset_parameters(init_type=init_type, init_enc_alpha=init_enc_alpha, init_dec_alpha=init_dec_alpha)
        if lang_embs is not None:
            torch.nn.init.normal_(self.encoder.language_embedding.weight, mean=0, std=attention_dimension ** -0.5)

        # define criterion
        self.criterion = FastSpeech2Loss(use_masking=use_masking, use_weighted_masking=use_weighted_masking,
                                         include_portaspeech_losses=True)

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
        glow_loss, \
        kl_loss = self._forward(text_tensors,
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

        # calculate loss
        l1_loss, ssim_loss, duration_loss, pitch_loss, energy_loss = self.criterion(after_outs=after_outs,
                                                                                    before_outs=before_outs,
                                                                                    d_outs=d_outs, p_outs=p_outs,
                                                                                    e_outs=e_outs, ys=gold_speech,
                                                                                    ds=gold_durations, ps=gold_pitch,
                                                                                    es=gold_energy,
                                                                                    ilens=text_lengths,
                                                                                    olens=speech_lengths)

        if return_mels:
            if after_outs is None:
                after_outs = before_outs
            return l1_loss, ssim_loss, duration_loss, pitch_loss, energy_loss, glow_loss, kl_loss, after_outs
        return l1_loss, ssim_loss, duration_loss, pitch_loss, energy_loss, glow_loss, kl_loss

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

        encoded_texts, _ = self.encoder(text_tensors, text_masks,
                                        utterance_embedding=utterance_embedding,
                                        lang_ids=lang_ids)  # (B, Tmax, adim)

        # forward duration predictor and variance predictors
        text_nonpadding_mask = make_non_pad_mask(text_lens, device=text_lens.device)

        if self.multilingual_model:
            lang_embs = self.encoder.language_embedding(lang_ids)
            utterance_embedding_for_variance_predictors = self.embedding_prenet_for_variance_predictors(
                torch.cat([utterance_embedding.detach(),
                           lang_embs.detach().squeeze(1)], dim=1))
        else:
            utterance_embedding_for_variance_predictors = utterance_embedding.detach()

        if not is_inference:
            pitch_z, loss_kl_pitch, _, _, _ = self.pitch_vae(gold_pitch,
                                                             nonpadding=text_nonpadding_mask.unsqueeze(1),
                                                             cond=encoded_texts.transpose(1, 2).detach(),
                                                             utt_emb=utterance_embedding_for_variance_predictors,
                                                             infer=is_inference)
            energy_z, loss_kl_energy, _, _, _ = self.energy_vae(gold_energy,
                                                                nonpadding=text_nonpadding_mask.unsqueeze(1),
                                                                cond=encoded_texts.transpose(1, 2).detach(),
                                                                utt_emb=utterance_embedding_for_variance_predictors,
                                                                infer=is_inference)
            duration_z, loss_kl_duration, _, _, _ = self.duration_vae(gold_durations.unsqueeze(-1).float(),
                                                                      nonpadding=text_nonpadding_mask.unsqueeze(1),
                                                                      cond=encoded_texts.transpose(1, 2).detach(),
                                                                      utt_emb=utterance_embedding_for_variance_predictors,
                                                                      infer=is_inference)

            min_dist = 0.2  # minimum distance between prior and posterior to avoid posterior collapse
            # https://openreview.net/forum?id=BJe0Gn0cY7
            kl_loss = (loss_kl_pitch - min_dist).abs() + \
                      (loss_kl_duration - min_dist).abs() + \
                      (loss_kl_energy - min_dist).abs()

        else:
            pitch_z = self.pitch_vae(cond=encoded_texts.transpose(1, 2),
                                     infer=is_inference)
            energy_z = self.energy_vae(cond=encoded_texts.transpose(1, 2),
                                       infer=is_inference)
            duration_z = self.duration_vae(cond=encoded_texts.transpose(1, 2),
                                           infer=is_inference)
            kl_loss = None

        pitch_predictions = self.pitch_vae.decoder(pitch_z,
                                                   nonpadding=text_nonpadding_mask.unsqueeze(1),
                                                   cond=encoded_texts.transpose(1, 2).detach(),
                                                   utt_emb=utterance_embedding_for_variance_predictors).transpose(1, 2)
        energy_predictions = self.energy_vae.decoder(energy_z,
                                                     nonpadding=text_nonpadding_mask.unsqueeze(1),
                                                     cond=encoded_texts.transpose(1, 2).detach(),
                                                     utt_emb=utterance_embedding_for_variance_predictors).transpose(1,
                                                                                                                    2)
        predicted_durations = self.duration_vae.decoder(duration_z,
                                                        nonpadding=text_nonpadding_mask.unsqueeze(1),
                                                        cond=encoded_texts.transpose(1, 2).detach(),
                                                        utt_emb=utterance_embedding_for_variance_predictors).squeeze(1)

        if is_inference:
            predicted_durations = make_estimated_durations_usable_for_inference(predicted_durations)
            embedded_pitch_curve = self.pitch_embed(pitch_predictions.transpose(1, 2)).transpose(1, 2)
            embedded_energy_curve = self.energy_embed(energy_predictions.transpose(1, 2)).transpose(1, 2)
            encoded_texts = encoded_texts + embedded_energy_curve + embedded_pitch_curve
            encoded_texts = self.length_regulator(encoded_texts, predicted_durations, alpha)  # (B, Lmax, adim)
        else:
            # use groundtruth in training
            embedded_pitch_curve = self.pitch_embed(gold_pitch.transpose(1, 2)).transpose(1, 2)
            embedded_energy_curve = self.energy_embed(gold_energy.transpose(1, 2)).transpose(1, 2)
            encoded_texts = encoded_texts + embedded_energy_curve + embedded_pitch_curve
            encoded_texts = self.length_regulator(encoded_texts, gold_durations)  # (B, Lmax, adim)

        # forward the decoder
        # (in PortaSpeech this is a VAE, but I believe this is a bit misplaced, and it is highly unstable)
        # (convolutions are sufficient, self-attention is not needed with the strong pitch and energy conditioning)
        # (the variance needs to happen at the level of the variance predictors, not here)
        if not is_inference:
            speech_nonpadding_mask = make_non_pad_mask(lengths=speech_lens,
                                                       device=speech_lens.device).unsqueeze(1).transpose(1, 2)
        else:
            speech_nonpadding_mask = None

        if speech_lens is not None and not is_inference:
            decoder_masks = self._source_mask(speech_lens)
        else:
            decoder_masks = None

        decoded_speech, _ = self.decoder(encoded_texts, decoder_masks, utterance_embedding)
        predicted_spectrogram_before_postnet = self.feat_out(decoded_speech).view(decoded_speech.size(0), -1, self.odim)

        predicted_spectrogram_after_postnet = None

        # forward flow post-net
        if run_glow:
            if is_inference:
                predicted_spectrogram_after_postnet = self.run_post_glow(tgt_mels=None,
                                                                         infer=is_inference,
                                                                         mel_out=predicted_spectrogram_before_postnet,
                                                                         encoded_texts=encoded_texts,
                                                                         tgt_nonpadding=None)
            else:
                glow_loss = self.run_post_glow(tgt_mels=gold_speech,
                                               infer=is_inference,
                                               mel_out=predicted_spectrogram_before_postnet.detach(),
                                               encoded_texts=encoded_texts.detach(),
                                               tgt_nonpadding=speech_nonpadding_mask.transpose(1, 2))
        else:
            glow_loss = torch.Tensor([0]).to(encoded_texts.device)

        if not is_inference:
            return predicted_spectrogram_before_postnet, predicted_spectrogram_after_postnet, predicted_durations, pitch_predictions, energy_predictions, glow_loss, kl_loss
        else:
            return predicted_spectrogram_before_postnet, predicted_spectrogram_after_postnet, predicted_durations, pitch_predictions, energy_predictions

    def inference(self,
                  text,
                  speech=None,
                  durations=None,
                  pitch=None,
                  energy=None,
                  alpha=1.0,
                  utterance_embedding=None,
                  return_duration_pitch_energy=False,
                  lang_id=None):
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
                                           lang_ids=lang_id)  # (1, L, odim)
        for phoneme_index, phoneme_vector in enumerate(xs.squeeze()):
            if phoneme_vector[get_feature_to_index_lookup()["voiced"]] == 0:
                pitch_predictions[0][phoneme_index] = 0.0
        self.train()
        if after_outs is None:
            after_outs = before_outs
        if return_duration_pitch_energy:
            return (before_outs[0], after_outs[0]), d_outs[0], pitch_predictions[0], energy_predictions[0]
        return after_outs[0]

    def run_post_glow(self, tgt_mels, infer, mel_out, encoded_texts, tgt_nonpadding, detach_postflow_input=True):
        x_recon = mel_out.transpose(1, 2)
        g = x_recon
        B, _, T = g.shape
        g = torch.cat([g, encoded_texts.transpose(1, 2)], 1)
        g = self.g_proj(g)
        prior_dist = self.prior_dist
        if not infer:
            y_lengths = tgt_nonpadding.sum(-1)
            if detach_postflow_input:
                g = g.detach()
            tgt_mels = tgt_mels.transpose(1, 2)
            z_postflow, ldj = self.post_flow(tgt_mels, tgt_nonpadding, g=g)
            ldj = ldj / y_lengths / 80
            postflow_loss = -prior_dist.log_prob(z_postflow).mean() - ldj.mean()
            if torch.isnan(postflow_loss):
                print("postflow loss is NaN, skipping postflow this step")
                return 0.0
            else:
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
