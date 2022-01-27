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
                 weights,
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
                 postnet_dropout_rate=0.5,
                 # additional features
                 utt_embed_dim=704,
                 connect_utt_emb_at_encoder_out=True,
                 lang_embs=100):
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
        self.encoder = Conformer(idim=idim, attention_dim=adim, attention_heads=aheads, linear_units=eunits, num_blocks=elayers,
                                 input_layer=embed, dropout_rate=transformer_enc_dropout_rate,
                                 positional_dropout_rate=transformer_enc_positional_dropout_rate, attention_dropout_rate=transformer_enc_attn_dropout_rate,
                                 normalize_before=encoder_normalize_before, concat_after=encoder_concat_after,
                                 positionwise_conv_kernel_size=positionwise_conv_kernel_size, macaron_style=use_macaron_style_in_conformer,
                                 use_cnn_module=use_cnn_in_conformer, cnn_module_kernel=conformer_enc_kernel_size, zero_triu=False,
                                 utt_embed=utt_embed_dim, connect_utt_emb_at_encoder_out=connect_utt_emb_at_encoder_out, lang_embs=lang_embs)
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
        self.load_state_dict(weights)

    def _forward(self, text_tensors, text_lens, gold_speech=None, speech_lens=None,
                 gold_durations=None, gold_pitch=None, gold_energy=None,
                 is_inference=False, alpha=1.0, utterance_embedding=None, lang_ids=None):
        # forward encoder
        text_masks = self._source_mask(text_lens)

        encoded_texts, _ = self.encoder(text_tensors, text_masks, utterance_embedding=utterance_embedding, lang_ids=lang_ids)  # (B, Tmax, adim)

        # forward duration predictor and variance predictors
        duration_masks = make_pad_mask(text_lens, device=text_lens.device)

        if self.stop_gradient_from_pitch_predictor:
            pitch_predictions = self.pitch_predictor(encoded_texts.detach(), duration_masks.unsqueeze(-1))
        else:
            pitch_predictions = self.pitch_predictor(encoded_texts, duration_masks.unsqueeze(-1))

        if self.stop_gradient_from_energy_predictor:
            energy_predictions = self.energy_predictor(encoded_texts.detach(), duration_masks.unsqueeze(-1))
        else:
            energy_predictions = self.energy_predictor(encoded_texts, duration_masks.unsqueeze(-1))

        if is_inference:
            if gold_durations is not None:
                duration_predictions = gold_durations
            else:
                duration_predictions = self.duration_predictor.inference(encoded_texts, duration_masks)
            if gold_pitch is not None:
                pitch_predictions = gold_pitch
            if gold_energy is not None:
                energy_predictions = gold_energy
            pitch_embeddings = self.pitch_embed(pitch_predictions.transpose(1, 2)).transpose(1, 2)
            energy_embeddings = self.energy_embed(energy_predictions.transpose(1, 2)).transpose(1, 2)
            encoded_texts = encoded_texts + energy_embeddings + pitch_embeddings
            encoded_texts = self.length_regulator(encoded_texts, duration_predictions, alpha)
        else:
            duration_predictions = self.duration_predictor(encoded_texts, duration_masks)

            # use groundtruth in training
            pitch_embeddings = self.pitch_embed(gold_pitch.transpose(1, 2)).transpose(1, 2)
            energy_embeddings = self.energy_embed(gold_energy.transpose(1, 2)).transpose(1, 2)
            encoded_texts = encoded_texts + energy_embeddings + pitch_embeddings
            encoded_texts = self.length_regulator(encoded_texts, gold_durations)  # (B, Lmax, adim)

        # forward decoder
        if speech_lens is not None and not is_inference:
            if self.reduction_factor > 1:
                olens_in = speech_lens.new([olen // self.reduction_factor for olen in speech_lens])
            else:
                olens_in = speech_lens
            h_masks = self._source_mask(olens_in)
        else:
            h_masks = None
        zs, _ = self.decoder(encoded_texts, h_masks)  # (B, Lmax, adim)
        before_outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)  # (B, Lmax, odim)

        # postnet -> (B, Lmax//r * r, odim)
        after_outs = before_outs + self.postnet(before_outs.transpose(1, 2)).transpose(1, 2)

        return before_outs, after_outs, duration_predictions, pitch_predictions, energy_predictions

    @torch.no_grad()
    def forward(self,
                text,
                speech=None,
                durations=None,
                pitch=None,
                energy=None,
                utterance_embedding=None,
                return_duration_pitch_energy=False,
                lang_id=None):
        """
        Generate the sequence of features given the sequences of characters.

        Args:
            text: Input sequence of characters
            speech: Feature sequence to extract style
            durations: Groundtruth of duration
            pitch: Groundtruth of token-averaged pitch
            energy: Groundtruth of token-averaged energy
            return_duration_pitch_energy: whether to return the list of predicted durations for nicer plotting
            utterance_embedding: embedding of utterance wide parameters

        Returns:
            Mel Spectrogram

        """
        self.eval()
        # setup batch axis
        ilens = torch.tensor([text.shape[0]], dtype=torch.long, device=text.device)
        if speech is not None:
            gold_speech = speech.unsqueeze(0)
        else:
            gold_speech = None
        if durations is not None:
            durations = durations.unsqueeze(0)
        if pitch is not None:
            pitch = pitch.unsqueeze(0)
        if energy is not None:
            energy = energy.unsqueeze(0)
        if lang_id is not None:
            lang_id = lang_id.unsqueeze(0)

        before_outs, after_outs, d_outs, pitch_predictions, energy_predictions = self._forward(text.unsqueeze(0),
                                                                                               ilens,
                                                                                               gold_speech=gold_speech,
                                                                                               gold_durations=durations,
                                                                                               is_inference=True,
                                                                                               gold_pitch=pitch,
                                                                                               gold_energy=energy,
                                                                                               utterance_embedding=utterance_embedding.unsqueeze(0),
                                                                                               lang_ids=lang_id)
        self.train()
        if return_duration_pitch_energy:
            return after_outs[0], d_outs[0], pitch_predictions[0], energy_predictions[0]
        return after_outs[0]

    def _source_mask(self, ilens):
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        return x_masks.unsqueeze(-2)
