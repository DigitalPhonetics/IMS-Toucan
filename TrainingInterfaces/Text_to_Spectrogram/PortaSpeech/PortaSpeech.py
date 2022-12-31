from abc import ABC

import torch
import torch.distributions as dist

from Layers.Conformer import Conformer
from Layers.LengthRegulator import LengthRegulator
from Preprocessing.articulatory_features import get_feature_to_index_lookup
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2Loss import FastSpeech2Loss
from TrainingInterfaces.Text_to_Spectrogram.PortaSpeech.FVAE import FVAE
from TrainingInterfaces.Text_to_Spectrogram.PortaSpeech.Glow import Glow
from TrainingInterfaces.Text_to_Spectrogram.PortaSpeech.PortaSpeechLayers import ConvBlocks
from Utility.utils import initialize
from Utility.utils import make_estimated_durations_usable_for_inference
from Utility.utils import make_non_pad_mask
from Utility.utils import make_pad_mask


class PortaSpeech(torch.nn.Module, ABC):
    """
    PortaSpeech 2 module, which is basically just a FastSpeech 2 module, but with an improved decoder and the PostNet is a flow.

    This is a module of FastSpeech 2 described in FastSpeech 2: Fast and
    High-Quality End-to-End Text to Speech. Instead of quantized pitch and
    energy, we use token-averaged value introduced in FastPitch: Parallel
    Text-to-speech with Pitch Prediction. The encoder and decoder are Conformers
    instead of regular Transformers.

        https://arxiv.org/abs/2006.04558
        https://arxiv.org/abs/2006.06873
        https://arxiv.org/pdf/2005.08100
        https://proceedings.neurips.cc/paper/2021/file/748d6b6ed8e13f857ceaa6cfbdca14b8-Paper.pdf

    """

    def __init__(self,
                 # network structure related
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
                 # encoder
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
        self.idim = idim
        self.odim = odim
        self.adim = adim
        self.eos = 1
        self.stop_gradient_from_pitch_predictor = stop_gradient_from_pitch_predictor
        self.stop_gradient_from_energy_predictor = stop_gradient_from_energy_predictor
        self.use_scaled_pos_enc = use_scaled_pos_enc
        self.multilingual_model = lang_embs is not None
        self.multispeaker_model = utt_embed_dim is not None

        # define encoder
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

        # initialize parameters
        self._reset_parameters(init_type=init_type, init_enc_alpha=init_enc_alpha, init_dec_alpha=init_dec_alpha)
        if lang_embs is not None:
            torch.nn.init.normal_(self.encoder.language_embedding.weight, mean=0, std=adim ** -0.5)

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
        d_masks = make_pad_mask(text_lens, device=text_lens.device)

        if not is_inference:
            pitch_z, loss_kl_pitch, _, _, _ = self.pitch_vae(gold_pitch,
                                                             nonpadding=~d_masks.unsqueeze(1),
                                                             cond=encoded_texts.transpose(1, 2).detach(),
                                                             utt_emb=utterance_embedding,
                                                             infer=is_inference)
            energy_z, loss_kl_energy, _, _, _ = self.energy_vae(gold_energy,
                                                                nonpadding=~d_masks.unsqueeze(1),
                                                                cond=encoded_texts.transpose(1, 2),
                                                                utt_emb=utterance_embedding,
                                                                infer=is_inference)
            duration_z, loss_kl_duration, _, _, _ = self.duration_vae(gold_durations.unsqueeze(-1).float(),
                                                                      nonpadding=~d_masks.unsqueeze(1),
                                                                      cond=encoded_texts.transpose(1, 2),
                                                                      utt_emb=utterance_embedding,
                                                                      infer=is_inference)
            kl_loss = torch.clamp(loss_kl_pitch + loss_kl_duration + loss_kl_energy, min=0.0)
        else:
            pitch_z = self.pitch_vae(cond=encoded_texts.transpose(1, 2),
                                     infer=is_inference)
            energy_z = self.energy_vae(cond=encoded_texts.transpose(1, 2),
                                       infer=is_inference)
            duration_z = self.duration_vae(cond=encoded_texts.transpose(1, 2),
                                           infer=is_inference)
            kl_loss = None

        pitch_predictions = self.pitch_vae.decoder(pitch_z,
                                                   nonpadding=~d_masks.unsqueeze(1),
                                                   cond=encoded_texts.transpose(1, 2).detach(),
                                                   utt_emb=utterance_embedding).transpose(1, 2)
        energy_predictions = self.energy_vae.decoder(energy_z,
                                                     nonpadding=~d_masks.unsqueeze(1),
                                                     cond=encoded_texts.transpose(1, 2),
                                                     utt_emb=utterance_embedding).transpose(1, 2)
        predicted_durations = self.duration_vae.decoder(duration_z,
                                                        nonpadding=~d_masks.unsqueeze(1),
                                                        cond=encoded_texts.transpose(1, 2),
                                                        utt_emb=utterance_embedding).squeeze(1)

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

        # forward the simple dilated convolutional decoder
        # (in PortaSpeech this is a VAE, but I believe this is a bit misplaced, and it is highly unstable)
        # (convolutions are sufficient, self-attention is not needed with the strong pitch and energy conditioning)
        # (the variance needs to happen at the level of the variance predictors, not here)
        if not is_inference:
            target_non_padding_mask = make_non_pad_mask(lengths=speech_lens,
                                                        device=speech_lens.device).unsqueeze(1).transpose(1, 2)
        else:
            target_non_padding_mask = None
        before_outs = self.decoder(encoded_texts, nonpadding=target_non_padding_mask,
                                   utt_emb=utterance_embedding).transpose(1, 2)
        before_outs = self.out_proj(before_outs).transpose(1, 2)

        after_outs = None

        # forward flow post-net
        if run_glow:
            if is_inference:
                after_outs = self.run_post_glow(tgt_mels=None,
                                                infer=is_inference,
                                                mel_out=before_outs,
                                                encoded_texts=encoded_texts,
                                                tgt_nonpadding=None)
            else:
                glow_loss = self.run_post_glow(tgt_mels=gold_speech,
                                               infer=is_inference,
                                               mel_out=before_outs,
                                               encoded_texts=encoded_texts,
                                               tgt_nonpadding=target_non_padding_mask.transpose(1, 2))
        else:
            glow_loss = torch.Tensor([0]).to(encoded_texts.device)

        if not is_inference:
            return before_outs, after_outs, predicted_durations, pitch_predictions, energy_predictions, glow_loss, kl_loss
        else:
            return before_outs, after_outs, predicted_durations, pitch_predictions, energy_predictions

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
        """
        Make masks for self-attention.

        Args:
            ilens (LongTensor): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor for self-attention.

        """
        x_masks = make_non_pad_mask(ilens, device=ilens.device)
        return x_masks.unsqueeze(-2)

    def _reset_parameters(self, init_type, init_enc_alpha, init_dec_alpha):
        # initialize parameters
        if init_type != "pytorch":
            initialize(self, init_type)
