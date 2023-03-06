from abc import ABC

import torch
from torch.nn import LayerNorm
from torch.nn import Linear
from torch.nn import Sequential
from torch.nn import Tanh

from Layers.Conformer import Conformer
from Layers.LengthRegulator import LengthRegulator
from Preprocessing.articulatory_features import get_feature_to_index_lookup
from TrainingInterfaces.Text_to_Spectrogram.ToucanTTS.Glow import Glow
from TrainingInterfaces.Text_to_Spectrogram.ToucanTTS.StochasticVariancePredictor import StochasticVariancePredictor
from TrainingInterfaces.Text_to_Spectrogram.ToucanTTS.ToucanTTSLoss import ToucanTTSLoss
from Utility.utils import initialize
from Utility.utils import make_non_pad_mask


class ToucanTTS(torch.nn.Module, ABC):
    """
    ToucanTTS module, which is basically just a FastSpeech 2 module,
    but with stochastic variance predictors inspired by the flow based
    duration predictor in VITS. The PostNet is also a normalizing flow,
    like in PortaSpeech. Furthermore, the pitch and energy values are
    averaged per-phone, as in FastPitch to enable great controllability.
    The encoder and decoder are Conformers. The final output receives a
    WGAN discriminator feedback signal. Input features are configurations
    of the articulatory tract rather than phoneme identities. Speaker
    embedding conditioning is derived from GST and Adaspeech 4. There is
    a large focus on multilinguality through numerous designs, such as
    explicit word boundaries.

    TODO WGAN objective
    TODO stochastic variance predictors
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
                 # energy predictor
                 energy_embed_kernel_size=1,
                 energy_embed_dropout=0.0,
                 # pitch predictor
                 pitch_embed_kernel_size=1,
                 pitch_embed_dropout=0.0,
                 # training related
                 transformer_enc_dropout_rate=0.2,
                 transformer_enc_positional_dropout_rate=0.2,
                 transformer_enc_attn_dropout_rate=0.2,
                 transformer_dec_dropout_rate=0.2,
                 transformer_dec_positional_dropout_rate=0.2,
                 transformer_dec_attn_dropout_rate=0.2,
                 init_type="xavier_uniform",
                 init_enc_alpha=1.0,
                 init_dec_alpha=1.0,
                 # additional features
                 utt_embed_dim=64,
                 detach_postflow=False,
                 lang_embs=8000):
        super().__init__()

        # store hyperparameters
        self.idim = input_feature_dimensions
        self.odim = output_spectrogram_channels
        self.adim = attention_dimension
        self.eos = 1
        self.detach_postflow = detach_postflow
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

        self.duration_predictor = StochasticVariancePredictor(in_channels=attention_dimension,
                                                              kernel_size=3,
                                                              p_dropout=0.5,
                                                              n_flows=4,
                                                              gin_channels=utt_embed_dim)

        self.pitch_predictor = StochasticVariancePredictor(in_channels=attention_dimension,
                                                           kernel_size=3,
                                                           p_dropout=0.5,
                                                           n_flows=4,
                                                           gin_channels=utt_embed_dim)

        self.energy_predictor = StochasticVariancePredictor(in_channels=attention_dimension,
                                                            kernel_size=3,
                                                            p_dropout=0.5,
                                                            n_flows=4,
                                                            gin_channels=utt_embed_dim)

        self.pitch_embed = Sequential(
            torch.nn.Conv1d(in_channels=1,
                            out_channels=attention_dimension,
                            kernel_size=pitch_embed_kernel_size,
                            padding=(pitch_embed_kernel_size - 1) // 2),
            torch.nn.Dropout(pitch_embed_dropout))

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
        if self.multispeaker_model:
            self.decoder_in_embedding_projection = Sequential(Linear(attention_dimension + utt_embed_dim, attention_dimension), LayerNorm(attention_dimension))
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
            sigmoid_scale=False,  # sigmoid_scale
            g_proj=torch.nn.Conv1d(output_spectrogram_channels + attention_dimension, gin_channels, 5, padding=2)
        )

        # initialize parameters
        self._reset_parameters(init_type=init_type, init_enc_alpha=init_enc_alpha, init_dec_alpha=init_dec_alpha)
        if lang_embs is not None:
            torch.nn.init.normal_(self.encoder.language_embedding.weight, mean=0, std=attention_dimension ** -0.5)

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
        Args:
            return_mels (Boolean): whether to return the predicted spectrogram
            text_tensors (LongTensor): Batch of padded text vectors (B, Tmax).
            text_lengths (LongTensor): Batch of lengths of each input (B,).
            gold_speech (Tensor): Batch of padded target features (B, Lmax, odim).
            speech_lengths (LongTensor): Batch of the lengths of each target (B,).
            gold_durations (LongTensor): Batch of padded durations (B, Tmax + 1).
            gold_pitch (Tensor): Batch of padded token-averaged pitch (B, Tmax + 1, 1).
            gold_energy (Tensor): Batch of padded token-averaged energy (B, Tmax + 1, 1).
            run_glow (Boolean): Whether to run the PostNet. There should be a warmup phase in the beginning.
            lang_ids (LongTensor): The language IDs used to access the language embedding table, if the model is multilingual
            utterance_embedding (Tensor): Batch of embeddings to condition the TTS on, if the model is multispeaker
        """
        before_outs, \
        after_outs, \
        duration_loss, \
        pitch_loss, \
        energy_loss, \
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

        # calculate loss
        l1_loss = self.criterion(after_outs=None,
                                 # if a regular PostNet is used, the post-PostNet outs have to go here. The flow has its own loss though, so we hard-code this to None
                                 before_outs=before_outs,
                                 ys=gold_speech,
                                 olens=speech_lengths)

        if return_mels:
            if after_outs is None:
                after_outs = before_outs
            return l1_loss, duration_loss, pitch_loss, energy_loss, glow_loss, after_outs
        return l1_loss, duration_loss, pitch_loss, energy_loss, glow_loss

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

        encoded_texts, _ = self.encoder(text_tensors, text_masks, utterance_embedding=utterance_embedding, lang_ids=lang_ids)

        if self.multispeaker_model:
            utterance_embedding_expanded = utterance_embedding.unsqueeze(-1)
        else:
            utterance_embedding_expanded = None

        if is_inference:
            # predicting pitch, energy and duration. All predictions are made in log space, so we apply exp to them.
            pitch_predictions = self.pitch_predictor(encoded_texts.transpose(1, 2), text_masks, w=None, g=utterance_embedding_expanded, reverse=True)

            for phoneme_index, phoneme_vector in enumerate(text_tensors.squeeze()):
                if phoneme_vector[get_feature_to_index_lookup()["voiced"]] == 0:
                    pitch_predictions[0][0][phoneme_index] = 0.0

            embedded_pitch_curve = self.pitch_embed(pitch_predictions).transpose(1, 2)

            energy_predictions = self.energy_predictor(encoded_texts.transpose(1, 2), text_masks, w=None, g=utterance_embedding_expanded, reverse=True)
            embedded_energy_curve = self.energy_embed(energy_predictions).transpose(1, 2)

            predicted_durations = self.duration_predictor(encoded_texts.transpose(1, 2), text_masks, w=None, g=utterance_embedding_expanded, reverse=True)
            predicted_durations = torch.ceil(torch.exp(predicted_durations)).long()

            for phoneme_index, phoneme_vector in enumerate(text_tensors.squeeze()):
                if phoneme_vector[get_feature_to_index_lookup()["word-boundary"]] == 1:
                    predicted_durations[0][0][phoneme_index] = 0

            encoded_texts = encoded_texts + embedded_pitch_curve + embedded_energy_curve

            upsampled_enriched_encoded_texts = self.length_regulator(encoded_texts, predicted_durations.squeeze(0), alpha)

        else:
            # training with teacher forcing
            idx = gold_pitch != 0  # once more thanks to ptrblck https://discuss.pytorch.org/t/calculating-logarithm-of-non-zero-values-in-pytorch/39303
            scaled_pitch_targets = gold_pitch.detach()
            scaled_pitch_targets[idx] = torch.exp(gold_pitch[idx])  # we scale up, so that the log in the flow can handle the value ranges better.
            pitch_flow_loss = torch.sum(self.pitch_predictor(encoded_texts.transpose(1, 2), text_masks, w=scaled_pitch_targets.transpose(1, 2), g=utterance_embedding_expanded, reverse=False))
            pitch_flow_loss = torch.sum(pitch_flow_loss / torch.sum(text_masks))  # weighted masking
            embedded_pitch_curve = self.pitch_embed(gold_pitch.transpose(1, 2)).transpose(1, 2)

            scaled_energy_targets = gold_energy.detach()
            scaled_energy_targets[idx] = torch.exp(gold_energy[idx])  # we scale up, so that the log in the flow can handle the value ranges better.
            energy_flow_loss = torch.sum(self.energy_predictor(encoded_texts.transpose(1, 2), text_masks, w=scaled_energy_targets.transpose(1, 2), g=utterance_embedding_expanded, reverse=False))
            energy_flow_loss = torch.sum(energy_flow_loss / torch.sum(text_masks))  # weighted masking
            embedded_energy_curve = self.energy_embed(gold_energy.transpose(1, 2)).transpose(1, 2)

            duration_flow_loss = self.duration_predictor(encoded_texts.transpose(1, 2), text_masks, w=gold_durations.float().unsqueeze(1), g=utterance_embedding_expanded, reverse=False)
            duration_flow_loss = torch.sum(duration_flow_loss / torch.sum(text_masks))  # weighted masking

            encoded_texts = encoded_texts + embedded_pitch_curve + embedded_energy_curve

            upsampled_enriched_encoded_texts = self.length_regulator(encoded_texts, gold_durations, alpha)

        # forward the decoder
        decoder_masks = self._source_mask(speech_lens) if speech_lens is not None and not is_inference else None
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
                predicted_spectrogram_after_postnet = self.post_flow(tgt_mels=None,
                                                                     infer=is_inference,
                                                                     mel_out=before_enriched,
                                                                     encoded_texts=upsampled_enriched_encoded_texts,
                                                                     tgt_nonpadding=None).squeeze()

            else:
                glow_loss = self.post_flow(tgt_mels=gold_speech,
                                           infer=is_inference,
                                           mel_out=before_enriched,
                                           encoded_texts=upsampled_enriched_encoded_texts,
                                           tgt_nonpadding=decoder_masks)
        if is_inference:
            return predicted_spectrogram_before_postnet.squeeze(), \
                   predicted_spectrogram_after_postnet, \
                   predicted_durations.squeeze(), \
                   pitch_predictions.squeeze(), \
                   energy_predictions.squeeze()
        else:
            return predicted_spectrogram_before_postnet, \
                   predicted_spectrogram_after_postnet, \
                   duration_flow_loss * 0.1, \
                   pitch_flow_loss * 0.1, \
                   energy_flow_loss * 0.1, \
                   glow_loss

    def inference(self,
                  text,
                  speech=None,
                  alpha=1.0,
                  utterance_embedding=None,
                  return_duration_pitch_energy=False,
                  lang_id=None,
                  run_postflow=True):
        """
        Args:
            text (LongTensor): Input sequence of characters (T,).
            speech (Tensor, optional): Feature sequence to extract style (N, idim).
            alpha (float, optional): Alpha to control the speed.
            return_duration_pitch_energy (Boolean): whether to return the list of predicted durations for nicer plotting
            run_postflow (Boolean): Whether to run the PostNet. There should be a warmup phase in the beginning.
            lang_id (LongTensor): The language ID used to access the language embedding table, if the model is multilingual
            utterance_embedding (Tensor): Embedding to condition the TTS on, if the model is multispeaker
        """
        self.eval()
        x, y = text, speech

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
        self.train()
        if after_outs is None:
            after_outs = before_outs
        if return_duration_pitch_energy:
            return (before_outs, after_outs), d_outs, pitch_predictions, energy_predictions
        return after_outs

    def _source_mask(self, ilens):
        # Make masks for self-attention.
        x_masks = make_non_pad_mask(ilens, device=ilens.device)
        return x_masks.unsqueeze(-2)

    def _reset_parameters(self, init_type, init_enc_alpha, init_dec_alpha):
        # initialize parameters
        if init_type != "pytorch":
            initialize(self, init_type)


def _integrate_with_utt_embed(hs, utt_embeddings, projection):
    # concat hidden states with spk embeds and then apply projection
    embeddings_expanded = torch.nn.functional.normalize(utt_embeddings).unsqueeze(1).expand(-1, hs.size(1), -1)
    hs = projection(torch.cat([hs, embeddings_expanded], dim=-1))
    return hs


if __name__ == '__main__':
    print(sum(p.numel() for p in ToucanTTS().parameters() if p.requires_grad))

    print(" TESTING TRAINING ")

    dummy_text_batch = torch.randint(low=0, high=2, size=[3, 3, 62]).float()  # [Batch, Sequence Length, Features per Phone]
    dummy_text_lens = torch.LongTensor([2, 3, 3])

    dummy_speech_batch = torch.randn([3, 30, 80])  # [Batch, Sequence Length, Spectrogram Buckets]
    dummy_speech_lens = torch.LongTensor([10, 30, 20])

    dummy_durations = torch.LongTensor([[10, 0, 0], [10, 15, 5], [5, 5, 10]])
    dummy_pitch = torch.Tensor([[[1.0], [0.], [0.]], [[1.1], [1.2], [0.8]], [[1.1], [1.2], [0.8]]])
    dummy_energy = torch.tensor([[[1.0], [0.], [0.]], [[1.1], [1.2], [0.8]], [[1.1], [1.2], [0.8]]])

    dummy_utterance_embed = torch.randn([3, 64])  # [Batch, Dimensions of Speaker Embedding]
    dummy_language_id = torch.LongTensor([5, 3, 2]).unsqueeze(1)

    model = ToucanTTS()
    l1, dl, pl, el, gl = model(dummy_text_batch,
                               dummy_text_lens,
                               dummy_speech_batch,
                               dummy_speech_lens,
                               dummy_durations,
                               dummy_pitch,
                               dummy_energy,
                               utterance_embedding=dummy_utterance_embed,
                               lang_ids=dummy_language_id)

    loss = l1 + gl + dl + el + pl
    print(loss)
    loss.backward()

    print(" TESTING INFERENCE ")
    dummy_text_batch = torch.randint(low=0, high=2, size=[12, 62]).float()  # [Sequence Length, Features per Phone]
    dummy_utterance_embed = torch.randn([64])  # [Dimensions of Speaker Embedding]
    dummy_language_id = torch.LongTensor([2])
    print(ToucanTTS().inference(dummy_text_batch,
                                utterance_embedding=dummy_utterance_embed,
                                lang_id=dummy_language_id).shape)
