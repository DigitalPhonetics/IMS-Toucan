import dotwiz
import torch
from torch.nn import Linear
from torch.nn import Sequential
from torch.nn import Tanh

from Layers.Conformer import Conformer
from Layers.DurationPredictor import DurationPredictor
from Layers.LengthRegulator import LengthRegulator
from Layers.VariancePredictor import VariancePredictor
from Preprocessing.articulatory_features import get_feature_to_index_lookup
from TTSTrainingInterfaces.ToucanTTS.wavenet import WN
from Utility.utils import make_non_pad_mask


class ToucanTTS(torch.nn.Module):

    def __init__(self,
                 weights,
                 config):
        super().__init__()

        self.config = config
        config = dotwiz.DotWiz(config)

        input_feature_dimensions = config.input_feature_dimensions
        attention_dimension = config.attention_dimension
        attention_heads = config.attention_heads
        positionwise_conv_kernel_size = config.positionwise_conv_kernel_size
        use_scaled_positional_encoding = config.use_scaled_positional_encoding
        use_macaron_style_in_conformer = config.use_macaron_style_in_conformer
        use_cnn_in_conformer = config.use_cnn_in_conformer
        encoder_layers = config.encoder_layers
        encoder_units = config.encoder_units
        encoder_normalize_before = config.encoder_normalize_before
        encoder_concat_after = config.encoder_concat_after
        conformer_encoder_kernel_size = config.conformer_encoder_kernel_size
        transformer_enc_dropout_rate = config.transformer_enc_dropout_rate
        transformer_enc_positional_dropout_rate = config.transformer_enc_positional_dropout_rate
        transformer_enc_attn_dropout_rate = config.transformer_enc_attn_dropout_rate
        decoder_layers = config.decoder_layers
        decoder_units = config.decoder_units
        decoder_concat_after = config.decoder_concat_after
        conformer_decoder_kernel_size = config.conformer_decoder_kernel_size
        decoder_normalize_before = config.decoder_normalize_before
        transformer_dec_dropout_rate = config.transformer_dec_dropout_rate
        transformer_dec_positional_dropout_rate = config.transformer_dec_positional_dropout_rate
        transformer_dec_attn_dropout_rate = config.transformer_dec_attn_dropout_rate
        duration_predictor_layers = config.duration_predictor_layers
        duration_predictor_kernel_size = config.duration_predictor_kernel_size
        duration_predictor_dropout_rate = config.duration_predictor_dropout_rate
        pitch_predictor_layers = config.pitch_predictor_layers
        pitch_predictor_kernel_size = config.pitch_predictor_kernel_size
        pitch_predictor_dropout = config.pitch_predictor_dropout
        pitch_embed_kernel_size = config.pitch_embed_kernel_size
        pitch_embed_dropout = config.pitch_embed_dropout
        energy_predictor_layers = config.energy_predictor_layers
        energy_predictor_kernel_size = config.energy_predictor_kernel_size
        energy_predictor_dropout = config.energy_predictor_dropout
        energy_embed_kernel_size = config.energy_embed_kernel_size
        energy_embed_dropout = config.energy_embed_dropout
        utt_embed_dim = config.utt_embed_dim
        lang_embs = config.lang_embs
        use_conditional_layernorm_embedding_integration = config.use_conditional_layernorm_embedding_integration
        num_codebooks = config.num_codebooks
        codebook_size = config.codebook_size
        use_wavenet_postnet = config.use_wavenet_postnet
        backtranslation_dim = config.backtranslation_dim

        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.input_feature_dimensions = input_feature_dimensions
        self.attention_dimension = attention_dimension
        self.use_scaled_pos_enc = use_scaled_positional_encoding
        self.multilingual_model = lang_embs is not None
        self.multispeaker_model = utt_embed_dim is not None
        self.use_wavenet_postnet = use_wavenet_postnet

        articulatory_feature_embedding = Sequential(Linear(input_feature_dimensions, 100), Tanh(), Linear(100, attention_dimension))
        self.encoder = Conformer(conformer_type="encoder",
                                 attention_dim=attention_dimension,
                                 attention_heads=attention_heads,
                                 linear_units=encoder_units,
                                 num_blocks=encoder_layers,
                                 input_layer=articulatory_feature_embedding,
                                 dropout_rate=transformer_enc_dropout_rate,
                                 positional_dropout_rate=transformer_enc_positional_dropout_rate,
                                 attention_dropout_rate=transformer_enc_attn_dropout_rate,
                                 normalize_before=encoder_normalize_before,
                                 concat_after=encoder_concat_after,
                                 positionwise_conv_kernel_size=positionwise_conv_kernel_size,
                                 macaron_style=use_macaron_style_in_conformer,
                                 use_cnn_module=False,
                                 cnn_module_kernel=conformer_encoder_kernel_size,
                                 zero_triu=False,
                                 utt_embed=utt_embed_dim,
                                 lang_embs=lang_embs,
                                 use_output_norm=True,
                                 use_conditional_layernorm_embedding_integration=use_conditional_layernorm_embedding_integration)

        self.duration_predictor = DurationPredictor(idim=attention_dimension,
                                                    n_layers=duration_predictor_layers,
                                                    n_chans=attention_dimension,
                                                    kernel_size=duration_predictor_kernel_size,
                                                    dropout_rate=duration_predictor_dropout_rate,
                                                    utt_embed_dim=utt_embed_dim,
                                                    use_conditional_layernorm_embedding_integration=use_conditional_layernorm_embedding_integration)

        self.pitch_predictor = VariancePredictor(idim=attention_dimension,
                                                 n_layers=pitch_predictor_layers,
                                                 n_chans=attention_dimension,
                                                 kernel_size=pitch_predictor_kernel_size,
                                                 dropout_rate=pitch_predictor_dropout,
                                                 utt_embed_dim=utt_embed_dim,
                                                 use_conditional_layernorm_embedding_integration=use_conditional_layernorm_embedding_integration)

        self.energy_predictor = VariancePredictor(idim=attention_dimension,
                                                  n_layers=energy_predictor_layers,
                                                  n_chans=attention_dimension,
                                                  kernel_size=energy_predictor_kernel_size,
                                                  dropout_rate=energy_predictor_dropout,
                                                  utt_embed_dim=utt_embed_dim,
                                                  use_conditional_layernorm_embedding_integration=use_conditional_layernorm_embedding_integration)

        self.pitch_embed = Sequential(torch.nn.Conv1d(in_channels=1,
                                                      out_channels=attention_dimension,
                                                      kernel_size=pitch_embed_kernel_size,
                                                      padding=(pitch_embed_kernel_size - 1) // 2),
                                      torch.nn.Dropout(pitch_embed_dropout))

        self.energy_embed = Sequential(torch.nn.Conv1d(in_channels=1,
                                                       out_channels=attention_dimension,
                                                       kernel_size=energy_embed_kernel_size,
                                                       padding=(energy_embed_kernel_size - 1) // 2),
                                       torch.nn.Dropout(energy_embed_dropout))

        self.length_regulator = LengthRegulator()

        self.decoder = Conformer(conformer_type="decoder",
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
                                 use_output_norm=False,
                                 utt_embed=utt_embed_dim,
                                 use_conditional_layernorm_embedding_integration=use_conditional_layernorm_embedding_integration)
        if self.use_wavenet_postnet:
            self.wn = WN(hidden_size=attention_dimension,
                         kernel_size=3,
                         dilation_rate=2,
                         n_layers=8,
                         c_cond=attention_dimension,
                         p_dropout=0.1,
                         share_cond_layers=False,
                         is_BTC=False,
                         use_weightnorm=True)

        self.hierarchical_classifier = torch.nn.ModuleList()
        self.backtranslation_heads = torch.nn.ModuleList()
        self.padding_id = self.codebook_size + 5
        for head in range(self.num_codebooks):
            self.hierarchical_classifier.append(torch.nn.Sequential(torch.nn.Dropout(0.),
                                                                    torch.nn.Linear(attention_dimension + head * backtranslation_dim, attention_dimension),
                                                                    torch.nn.Tanh(),
                                                                    torch.nn.Dropout(0.),
                                                                    torch.nn.Linear(attention_dimension, self.codebook_size)))
            self.backtranslation_heads.append(torch.nn.Embedding(num_embeddings=self.padding_id + 1, embedding_dim=backtranslation_dim, padding_idx=self.padding_id))

        self.load_state_dict(weights)
        self.eval()

    def _forward(self,
                 text_tensors,
                 text_lengths,
                 gold_durations=None,
                 gold_pitch=None,
                 gold_energy=None,
                 duration_scaling_factor=1.0,
                 utterance_embedding=None,
                 lang_ids=None,
                 pitch_variance_scale=1.0,
                 energy_variance_scale=1.0,
                 pause_duration_scaling_factor=1.0):

        if not self.multilingual_model:
            lang_ids = None

        if not self.multispeaker_model:
            utterance_embedding = None
        else:
            utterance_embedding = torch.nn.functional.normalize(utterance_embedding)

        # encoding the texts
        text_masks = make_non_pad_mask(text_lengths, device=text_lengths.device).unsqueeze(-2)
        encoded_texts, _ = self.encoder(text_tensors, text_masks, utterance_embedding=utterance_embedding, lang_ids=lang_ids)

        # predicting pitch, energy and durations
        pitch_predictions = self.pitch_predictor(encoded_texts, padding_mask=None, utt_embed=utterance_embedding) if gold_pitch is None else gold_pitch
        energy_predictions = self.energy_predictor(encoded_texts, padding_mask=None, utt_embed=utterance_embedding) if gold_energy is None else gold_energy
        predicted_durations = self.duration_predictor.inference(encoded_texts, padding_mask=None, utt_embed=utterance_embedding) if gold_durations is None else gold_durations

        # modifying the predictions with control parameters
        for phoneme_index, phoneme_vector in enumerate(text_tensors.squeeze(0)):
            if phoneme_vector[get_feature_to_index_lookup()["word-boundary"]] == 1:
                predicted_durations[0][phoneme_index] = 0
            if phoneme_vector[get_feature_to_index_lookup()["silence"]] == 1 and pause_duration_scaling_factor != 1.0:
                predicted_durations[0][phoneme_index] = torch.round(predicted_durations[0][phoneme_index].float() * pause_duration_scaling_factor).long()
        if duration_scaling_factor != 1.0:
            assert duration_scaling_factor > 0
            predicted_durations = torch.round(predicted_durations.float() * duration_scaling_factor).long()
        pitch_predictions = _scale_variance(pitch_predictions, pitch_variance_scale)
        energy_predictions = _scale_variance(energy_predictions, energy_variance_scale)

        # enriching the text with pitch and energy info
        embedded_pitch_curve = self.pitch_embed(pitch_predictions.transpose(1, 2)).transpose(1, 2)
        embedded_energy_curve = self.energy_embed(energy_predictions.transpose(1, 2)).transpose(1, 2)
        enriched_encoded_texts = encoded_texts + embedded_pitch_curve + embedded_energy_curve

        # predicting durations for text and upsampling accordingly
        upsampled_enriched_encoded_texts = self.length_regulator(enriched_encoded_texts, predicted_durations)

        # decoding spectrogram
        decoded_speech, _ = self.decoder(upsampled_enriched_encoded_texts, None, utterance_embedding=utterance_embedding)

        if self.use_wavenet_postnet:
            decoded_speech = decoded_speech + self.wn(x=decoded_speech.transpose(1, 2), nonpadding=None, cond=upsampled_enriched_encoded_texts.transpose(1, 2)).transpose(1, 2)

        # The codebooks are hierarchical: The first influences the second, but the second not the first.
        # This is because they are residual vector quantized, which makes them extremely space efficient
        # with just a few discrete tokens, but terribly difficult to predict.
        predicted_indexes = list()
        backtranslated_indexes = list()
        backtranslated_indexes.append(decoded_speech)
        for head_index, classifier_head in enumerate(self.hierarchical_classifier):
            # each codebook considers all previous codebooks.
            predicted_indexes.append(classifier_head(torch.cat(backtranslated_indexes, dim=2)))
            predicted_lookup_index = torch.argmax(predicted_indexes[-1], dim=-1)
            backtranslation = self.backtranslation_heads[head_index](predicted_lookup_index)
            if len(backtranslation.size()) == 1:
                backtranslation = backtranslation.unsqueeze(0)
            backtranslated_indexes.append(backtranslation)

        indexes = torch.cat(predicted_indexes, dim=2)
        # [Batch, Sequence, Hidden]
        indexes = indexes.view(decoded_speech.size(0), decoded_speech.size(1), self.num_codebooks, self.codebook_size)
        # [Batch, Sequence, Codebook, Classes]
        indexes = indexes.transpose(1, 2)
        # [Batch, Codebook, Sequence, Classes]
        indexes = indexes.transpose(2, 3)
        # [Batch, Codebook, Classes, Sequence]
        indexes = indexes.transpose(0, 1)
        # [Codebook, Batch, Classes, Sequence]

        return indexes, predicted_durations.squeeze(), pitch_predictions.squeeze(), energy_predictions.squeeze()

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
                pause_duration_scaling_factor=1.0):
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
            features spectrogram

        """
        # setup batch axis
        text_length = torch.tensor([text.shape[0]], dtype=torch.long, device=text.device)
        if durations is not None:
            durations = durations.unsqueeze(0).to(text.device)
        if pitch is not None:
            pitch = pitch.unsqueeze(0).to(text.device)
        if energy is not None:
            energy = energy.unsqueeze(0).to(text.device)
        if lang_id is not None:
            lang_id = lang_id.unsqueeze(0).to(text.device)

        outs, \
        predicted_durations, \
        pitch_predictions, \
        energy_predictions = self._forward(text.unsqueeze(0),
                                           text_length,
                                           gold_durations=durations,
                                           gold_pitch=pitch,
                                           gold_energy=energy,
                                           utterance_embedding=utterance_embedding.unsqueeze(0) if utterance_embedding is not None else None, lang_ids=lang_id,
                                           duration_scaling_factor=duration_scaling_factor,
                                           pitch_variance_scale=pitch_variance_scale,
                                           energy_variance_scale=energy_variance_scale,
                                           pause_duration_scaling_factor=pause_duration_scaling_factor)

        outs_indexed = list()
        for out in outs:
            outs_indexed.append(torch.argmax(out.squeeze(), dim=0))

        outs = torch.stack(outs_indexed)

        if return_duration_pitch_energy:
            return outs, predicted_durations, pitch_predictions, energy_predictions
        return outs

    def store_inverse_all(self):
        def remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(remove_weight_norm)


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
