import torch
from torch.nn import Linear
from torch.nn import Sequential
from torch.nn import Tanh

from Layers.Conformer import Conformer
from Layers.DurationPredictor import DurationPredictor
from Layers.LengthRegulator import LengthRegulator
from Layers.VariancePredictor import VariancePredictor
from Preprocessing.articulatory_features import get_feature_to_index_lookup
from TTSTrainingInterfaces.ToucanTTS.CodecRefinementTransformer import CodecRefinementTransformer
from TTSTrainingInterfaces.ToucanTTS.CodecRefinementTransformer import one_hot_sequence_to_token_sequence
from TTSTrainingInterfaces.ToucanTTS.ToucanTTSLoss import ToucanTTSLoss
from TTSTrainingInterfaces.ToucanTTS.wavenet import WN
from Utility.utils import initialize
from Utility.utils import make_non_pad_mask
from Utility.utils import make_pad_mask


class ToucanTTS(torch.nn.Module):
    """
    ToucanTTS module, which is mostly just a FastSpeech 2 module,
    but with lots of designs from different architectures accumulated
    and some major components added to put a large focus on multilinguality.

    Original contributions:
    - Inputs are configurations of the articulatory tract
    - Word boundaries are modeled explicitly in the encoder end removed before the decoder
    - Speaker embedding conditioning is derived from GST and Adaspeech 4
    - Responsiveness of variance predictors to utterance embedding is increased through conditional layer norm
    - The final output receives a GAN discriminator feedback signal

    Contributions inspired from elsewhere:
    - The PostNet is also a normalizing flow, like in PortaSpeech
    - Pitch and energy values are averaged per-phone, as in FastPitch to enable great controllability
    - The encoder and decoder are Conformers

    Things that were tried, but showed inferior performance so far:
    - Stochastic Duration Prediction
    - Stochastic Pitch Prediction
    - Stochastic Energy prediction
    """

    def __init__(self,
                 # network structure related
                 input_feature_dimensions=62,
                 attention_dimension=128,
                 attention_heads=4,
                 positionwise_conv_kernel_size=1,
                 use_scaled_positional_encoding=True,
                 init_type="xavier_uniform",
                 use_macaron_style_in_conformer=True,
                 use_cnn_in_conformer=True,

                 # encoder
                 encoder_layers=6,
                 encoder_units=1280,
                 encoder_normalize_before=True,
                 encoder_concat_after=False,
                 conformer_encoder_kernel_size=7,
                 transformer_enc_dropout_rate=0.1,
                 transformer_enc_positional_dropout_rate=0.1,
                 transformer_enc_attn_dropout_rate=0.1,

                 # decoder
                 decoder_layers=6,
                 decoder_units=1280,
                 decoder_concat_after=False,
                 conformer_decoder_kernel_size=31,  # 31 for spectrograms
                 decoder_normalize_before=True,
                 transformer_dec_dropout_rate=0.1,
                 transformer_dec_positional_dropout_rate=0.1,
                 transformer_dec_attn_dropout_rate=0.1,

                 # duration predictor
                 duration_predictor_layers=5,
                 duration_predictor_kernel_size=5,
                 duration_predictor_dropout_rate=0.2,

                 # pitch predictor
                 pitch_predictor_layers=5,
                 pitch_predictor_kernel_size=5,
                 pitch_predictor_dropout=0.3,
                 pitch_embed_kernel_size=1,
                 pitch_embed_dropout=0.0,

                 # energy predictor
                 energy_predictor_layers=2,
                 energy_predictor_kernel_size=3,
                 energy_predictor_dropout=0.5,
                 energy_embed_kernel_size=1,
                 energy_embed_dropout=0.0,

                 # additional features
                 utt_embed_dim=512,
                 lang_embs=8000,
                 use_conditional_layernorm_embedding_integration=False,
                 num_codebooks=5,  # between 1 and 9 when using the descript audio codec
                 codebook_size=1024,
                 backtranslation_dim=8,
                 use_wavenet_postnet=False,
                 use_language_model=False):
        super().__init__()

        self.config = {
            "input_feature_dimensions"                       : input_feature_dimensions,
            "attention_dimension"                            : attention_dimension,
            "attention_heads"                                : attention_heads,
            "positionwise_conv_kernel_size"                  : positionwise_conv_kernel_size,
            "use_scaled_positional_encoding"                 : use_scaled_positional_encoding,
            "init_type"                                      : init_type,
            "use_macaron_style_in_conformer"                 : use_macaron_style_in_conformer,
            "use_cnn_in_conformer"                           : use_cnn_in_conformer,
            "encoder_layers"                                 : encoder_layers,
            "encoder_units"                                  : encoder_units,
            "encoder_normalize_before"                       : encoder_normalize_before,
            "encoder_concat_after"                           : encoder_concat_after,
            "conformer_encoder_kernel_size"                  : conformer_encoder_kernel_size,
            "transformer_enc_dropout_rate"                   : transformer_enc_dropout_rate,
            "transformer_enc_positional_dropout_rate"        : transformer_enc_positional_dropout_rate,
            "transformer_enc_attn_dropout_rate"              : transformer_enc_attn_dropout_rate,
            "decoder_layers"                                 : decoder_layers,
            "decoder_units"                                  : decoder_units,
            "decoder_concat_after"                           : decoder_concat_after,
            "conformer_decoder_kernel_size"                  : conformer_decoder_kernel_size,
            "decoder_normalize_before"                       : decoder_normalize_before,
            "transformer_dec_dropout_rate"                   : transformer_dec_dropout_rate,
            "transformer_dec_positional_dropout_rate"        : transformer_dec_positional_dropout_rate,
            "transformer_dec_attn_dropout_rate"              : transformer_dec_attn_dropout_rate,
            "duration_predictor_layers"                      : duration_predictor_layers,
            "duration_predictor_kernel_size"                 : duration_predictor_kernel_size,
            "duration_predictor_dropout_rate"                : duration_predictor_dropout_rate,
            "pitch_predictor_layers"                         : pitch_predictor_layers,
            "pitch_predictor_kernel_size"                    : pitch_predictor_kernel_size,
            "pitch_predictor_dropout"                        : pitch_predictor_dropout,
            "pitch_embed_kernel_size"                        : pitch_embed_kernel_size,
            "pitch_embed_dropout"                            : pitch_embed_dropout,
            "energy_predictor_layers"                        : energy_predictor_layers,
            "energy_predictor_kernel_size"                   : energy_predictor_kernel_size,
            "energy_predictor_dropout"                       : energy_predictor_dropout,
            "energy_embed_kernel_size"                       : energy_embed_kernel_size,
            "energy_embed_dropout"                           : energy_embed_dropout,
            "utt_embed_dim"                                  : utt_embed_dim,
            "lang_embs"                                      : lang_embs,
            "use_conditional_layernorm_embedding_integration": use_conditional_layernorm_embedding_integration,
            "num_codebooks"                                  : num_codebooks,
            "codebook_size"                                  : codebook_size,
            "use_wavenet_postnet"                            : use_wavenet_postnet,
            "backtranslation_dim"                            : backtranslation_dim,
            "use_language_model"                             : use_language_model,
        }

        self.input_feature_dimensions = input_feature_dimensions
        self.attention_dimension = attention_dimension
        self.use_scaled_pos_enc = use_scaled_positional_encoding
        self.multilingual_model = lang_embs is not None
        self.multispeaker_model = utt_embed_dim is not None
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.use_wavenet_postnet = use_wavenet_postnet
        self.use_language_model = use_language_model

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
                                 use_cnn_module=use_cnn_in_conformer,
                                 cnn_module_kernel=conformer_encoder_kernel_size,
                                 zero_triu=False,
                                 utt_embed=utt_embed_dim,
                                 lang_embs=lang_embs,
                                 use_output_norm=True,
                                 use_conditional_layernorm_embedding_integration=use_conditional_layernorm_embedding_integration)

        self.duration_predictor = DurationPredictor(idim=attention_dimension, n_layers=duration_predictor_layers,
                                                    n_chans=attention_dimension,
                                                    kernel_size=duration_predictor_kernel_size,
                                                    dropout_rate=duration_predictor_dropout_rate,
                                                    utt_embed_dim=utt_embed_dim,
                                                    use_conditional_layernorm_embedding_integration=use_conditional_layernorm_embedding_integration)

        self.pitch_predictor = VariancePredictor(idim=attention_dimension, n_layers=pitch_predictor_layers,
                                                 n_chans=attention_dimension,
                                                 kernel_size=pitch_predictor_kernel_size,
                                                 dropout_rate=pitch_predictor_dropout,
                                                 utt_embed_dim=utt_embed_dim,
                                                 use_conditional_layernorm_embedding_integration=use_conditional_layernorm_embedding_integration)

        self.energy_predictor = VariancePredictor(idim=attention_dimension, n_layers=energy_predictor_layers,
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

        self.energy_embed = Sequential(torch.nn.Conv1d(in_channels=1, out_channels=attention_dimension, kernel_size=energy_embed_kernel_size,
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
            self.hierarchical_classifier.append(torch.nn.Sequential(torch.nn.Dropout(.2),
                                                                    torch.nn.Linear(attention_dimension + head * backtranslation_dim, attention_dimension),
                                                                    torch.nn.Tanh(),
                                                                    torch.nn.Dropout(.2),
                                                                    torch.nn.Linear(attention_dimension, self.codebook_size)))
            self.backtranslation_heads.append(torch.nn.Embedding(num_embeddings=self.padding_id + 1, embedding_dim=backtranslation_dim, padding_idx=self.padding_id))

        self.curriculum_state = 1

        if use_language_model:
            self.language_model = CodecRefinementTransformer(
                num_codebooks=num_codebooks,
                attention_dimension=attention_dimension,
                codebook_size=codebook_size,
                backtranslation_dim=backtranslation_dim,
                attention_heads=attention_heads,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
                use_macaron_style_in_conformer=use_macaron_style_in_conformer,
                use_cnn_in_conformer=use_cnn_in_conformer,
                decoder_layers=decoder_layers,
                decoder_units=decoder_units,
                decoder_concat_after=decoder_concat_after,
                conformer_decoder_kernel_size=conformer_decoder_kernel_size,
                decoder_normalize_before=decoder_normalize_before,
                transformer_dec_dropout_rate=transformer_dec_dropout_rate,
                transformer_dec_positional_dropout_rate=transformer_dec_positional_dropout_rate,
                transformer_dec_attn_dropout_rate=transformer_dec_attn_dropout_rate,
                utt_embed_dim=utt_embed_dim,
                use_conditional_layernorm_embedding_integration=use_conditional_layernorm_embedding_integration
            )

        # initialize parameters
        self._reset_parameters(init_type=init_type)
        if lang_embs is not None:
            torch.nn.init.normal_(self.encoder.language_embedding.weight, mean=0, std=attention_dimension ** -0.5)
        for backtranslation_head in self.backtranslation_heads:
            torch.nn.init.normal_(backtranslation_head.weight, mean=0, std=attention_dimension ** -0.5)

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
                return_feats=False,
                lang_ids=None,
                codebook_curriculum=None
                ):
        """
        Args:
            return_feats (Boolean): whether to return the predicted spectrogram
            text_tensors (LongTensor): Batch of padded text vectors (B, Tmax).
            text_lengths (LongTensor): Batch of lengths of each input (B,).
            gold_speech (Tensor): Batch of padded target features (B, Lmax, odim).
            speech_lengths (LongTensor): Batch of the lengths of each target (B,).
            gold_durations (LongTensor): Batch of padded durations (B, Tmax + 1).
            gold_pitch (Tensor): Batch of padded token-averaged pitch (B, Tmax + 1, 1).
            gold_energy (Tensor): Batch of padded token-averaged energy (B, Tmax + 1, 1).
            lang_ids (LongTensor): The language IDs used to access the language embedding table, if the model is multilingual
            utterance_embedding (Tensor): Batch of embeddings to condition the TTS on, if the model is multispeaker
            codebook_curriculum (Tensor): How many codebooks to use
        """
        outs, \
        predicted_durations, \
        predicted_pitch, \
        predicted_energy, \
        refiner_classification_loss, \
        mlm_loss = self._forward(text_tensors=text_tensors,
                                 text_lengths=text_lengths,
                                 gold_speech=gold_speech,
                                 speech_lengths=speech_lengths,
                                 gold_durations=gold_durations,
                                 gold_pitch=gold_pitch,
                                 gold_energy=gold_energy,
                                 utterance_embedding=utterance_embedding,
                                 is_inference=False,
                                 lang_ids=lang_ids,
                                 codebook_curriculum=codebook_curriculum)

        # calculate loss
        classification_loss, duration_loss, pitch_loss, energy_loss = self.criterion(predicted_features=outs,
                                                                                     gold_features=gold_speech,
                                                                                     features_lengths=speech_lengths,
                                                                                     text_lengths=text_lengths,
                                                                                     gold_durations=gold_durations,
                                                                                     predicted_durations=predicted_durations,
                                                                                     predicted_pitch=predicted_pitch,
                                                                                     predicted_energy=predicted_energy,
                                                                                     gold_pitch=gold_pitch,
                                                                                     gold_energy=gold_energy)

            
        if return_feats:
            return classification_loss, refiner_classification_loss, mlm_loss, duration_loss, pitch_loss, energy_loss, outs
        return classification_loss, refiner_classification_loss, mlm_loss, duration_loss, pitch_loss, energy_loss

    def _forward(self,
                 text_tensors,
                 text_lengths,
                 gold_speech=None,
                 speech_lengths=None,
                 gold_durations=None,
                 gold_pitch=None,
                 gold_energy=None,
                 is_inference=False,
                 utterance_embedding=None,
                 lang_ids=None,
                 codebook_curriculum=None):

        if not self.multilingual_model:
            lang_ids = None

        if not self.multispeaker_model:
            utterance_embedding = None
        else:
            utterance_embedding = torch.nn.functional.normalize(utterance_embedding)

        # encoding the texts
        text_masks = make_non_pad_mask(text_lengths, device=text_lengths.device).unsqueeze(-2)
        padding_masks = make_pad_mask(text_lengths, device=text_lengths.device)
        encoded_texts, _ = self.encoder(text_tensors, text_masks, utterance_embedding=utterance_embedding, lang_ids=lang_ids)

        if is_inference:
            # predicting pitch, energy and durations
            pitch_predictions = self.pitch_predictor(encoded_texts, padding_mask=None, utt_embed=utterance_embedding)
            energy_predictions = self.energy_predictor(encoded_texts, padding_mask=None, utt_embed=utterance_embedding)
            predicted_durations = self.duration_predictor.inference(encoded_texts, padding_mask=None, utt_embed=utterance_embedding)

            # modifying the predictions
            for phoneme_index, phoneme_vector in enumerate(text_tensors.squeeze(0)):
                if phoneme_vector[get_feature_to_index_lookup()["word-boundary"]] == 1:
                    predicted_durations[0][phoneme_index] = 0
            # enriching the text with pitch and energy info
            embedded_pitch_curve = self.pitch_embed(pitch_predictions.transpose(1, 2)).transpose(1, 2)
            embedded_energy_curve = self.energy_embed(energy_predictions.transpose(1, 2)).transpose(1, 2)
            enriched_encoded_texts = encoded_texts + embedded_pitch_curve + embedded_energy_curve

            # predicting durations for text and upsampling accordingly
            upsampled_enriched_encoded_texts = self.length_regulator(enriched_encoded_texts, predicted_durations)

        else:
            # training with teacher forcing
            pitch_predictions = self.pitch_predictor(encoded_texts, padding_mask=padding_masks.unsqueeze(-1), utt_embed=utterance_embedding)
            energy_predictions = self.energy_predictor(encoded_texts, padding_mask=padding_masks.unsqueeze(-1), utt_embed=utterance_embedding)
            predicted_durations = self.duration_predictor(encoded_texts, padding_mask=padding_masks, utt_embed=utterance_embedding)

            embedded_pitch_curve = self.pitch_embed(gold_pitch.transpose(1, 2)).transpose(1, 2)
            embedded_energy_curve = self.energy_embed(gold_energy.transpose(1, 2)).transpose(1, 2)
            enriched_encoded_texts = encoded_texts + embedded_energy_curve + embedded_pitch_curve

            upsampled_enriched_encoded_texts = self.length_regulator(enriched_encoded_texts, gold_durations)

        # decoding spectrogram
        decoder_masks = make_non_pad_mask(speech_lengths, device=speech_lengths.device).unsqueeze(-2) if speech_lengths is not None and not is_inference else None
        decoded_speech, _ = self.decoder(upsampled_enriched_encoded_texts, decoder_masks, utterance_embedding=utterance_embedding)

        if self.use_wavenet_postnet:
            decoded_speech = decoded_speech + self.wn(x=decoded_speech.transpose(1, 2), nonpadding=decoder_masks, cond=upsampled_enriched_encoded_texts.transpose(1, 2)).transpose(1, 2)

        # The codebooks are hierarchical: The first influences the second, but the second not the first.
        # This is because they are residual vector quantized, which makes them extremely space efficient
        # with just a few discrete tokens, but terribly difficult to predict.
        if not is_inference:
            # during training, we use teacher forcing to make it easier to predict the hierarchy of features.
            gold_indexes = list()
        predicted_indexes_one_hot = list()
        backtranslated_indexes = list()

        if codebook_curriculum is None or codebook_curriculum > self.num_codebooks:
            codebook_curriculum = self.num_codebooks
        if codebook_curriculum > self.curriculum_state:
            self.curriculum_state = codebook_curriculum
        for head_index, classifier_head in enumerate(self.hierarchical_classifier[:codebook_curriculum]):
            # each codebook considers all previous codebooks.
            if not is_inference:
                if len(gold_indexes) != 0:
                    decoded_speech = decoded_speech.detach()  # it is considered frozen for all heads except the first.
                predicted_indexes_one_hot.append(classifier_head(torch.cat([decoded_speech] + gold_indexes, dim=2)))
                gold_lookup_index = torch.argmax(gold_speech.transpose(0, 1)[head_index], dim=-1)
                gold_lookup_index = gold_lookup_index.masked_fill(mask=~decoder_masks.squeeze(1), value=self.padding_id)
                backtranslation = self.backtranslation_heads[head_index](gold_lookup_index)
                gold_indexes.append(backtranslation)
            else:
                predicted_indexes_one_hot.append(classifier_head(torch.cat([decoded_speech] + backtranslated_indexes, dim=2)))
                predicted_lookup_index = torch.argmax(predicted_indexes_one_hot[-1], dim=-1)
                backtranslation = self.backtranslation_heads[head_index](predicted_lookup_index)
                if len(backtranslation.size()) == 1:
                    backtranslation = backtranslation.unsqueeze(0)
                backtranslated_indexes.append(backtranslation)

        indexes = torch.cat(predicted_indexes_one_hot, dim=2)
        # [Batch, Sequence, Hidden]
        indexes = indexes.view(decoded_speech.size(0), decoded_speech.size(1), codebook_curriculum, self.codebook_size)
        # [Batch, Sequence, Codebook, Classes]
        indexes = indexes.transpose(1, 2)
        # [Batch, Codebook, Sequence, Classes]
        indexes = indexes.transpose(2, 3)
        # [Batch, Codebook, Classes, Sequence]
        indexes = indexes.transpose(0, 1)
        # [Codebook, Batch, Classes, Sequence]

        classification_loss, mlm_loss = None, None
        if self.use_language_model:
            if is_inference:
                if self.num_codebooks == codebook_curriculum:
                    indexes = self.language_model(index_sequence=one_hot_sequence_to_token_sequence(indexes), padding_mask=None, is_inference=is_inference, speaker_embedding=utterance_embedding, gold_index_sequence=None)
                else:
                    print("Skipping the language model, since the curriculum does not yet include all codebooks.")
            else:
                classification_loss, mlm_loss = self.language_model(index_sequence=one_hot_sequence_to_token_sequence(indexes), padding_mask=~decoder_masks.squeeze(1), is_inference=is_inference, speaker_embedding=utterance_embedding, gold_index_sequence=gold_speech)

        if is_inference:
            return indexes, \
                   predicted_durations.squeeze(), \
                   pitch_predictions.squeeze(), \
                   energy_predictions.squeeze()
        else:
            return indexes, \
                   predicted_durations, \
                   pitch_predictions, \
                   energy_predictions, \
                   classification_loss, \
                   mlm_loss

    @torch.inference_mode()
    def inference(self,
                  text,
                  speech=None,
                  utterance_embedding=None,
                  return_duration_pitch_energy=False,
                  lang_id=None):
        """
        Args:
            text (LongTensor): Input sequence of characters (T,).
            speech (Tensor, optional): Feature sequence to extract style (N, idim).
            return_duration_pitch_energy (Boolean): whether to return the list of predicted durations for nicer plotting
            lang_id (LongTensor): The language ID used to access the language embedding table, if the model is multilingual
            utterance_embedding (Tensor): Embedding to condition the TTS on, if the model is multispeaker
        """
        self.eval()

        # setup batch axis
        ilens = torch.tensor([text.shape[0]], dtype=torch.long, device=text.device)
        text_pseudobatched, speech_pseudobatched = text.unsqueeze(0), None
        if speech is not None:
            speech_pseudobatched = speech.unsqueeze(0)
        if lang_id is not None:
            lang_id = lang_id.unsqueeze(0)
        utterance_embeddings = utterance_embedding.unsqueeze(0) if utterance_embedding is not None else None

        outs, \
        duration_predictions, \
        pitch_predictions, \
        energy_predictions = self._forward(text_pseudobatched,
                                           ilens,
                                           speech_pseudobatched,
                                           is_inference=True,
                                           utterance_embedding=utterance_embeddings,
                                           lang_ids=lang_id,
                                           codebook_curriculum=self.curriculum_state)  # (1, L, odim)
        self.train()
        outs_indexed = list()
        for out in outs:
            outs_indexed.append(torch.argmax(out.squeeze(0), dim=0))

        outs = torch.stack(outs_indexed)
        if return_duration_pitch_energy:
            return outs, duration_predictions, pitch_predictions, energy_predictions
        return outs

    def _reset_parameters(self, init_type):
        # initialize parameters
        if init_type != "pytorch":
            initialize(self, init_type)


if __name__ == '__main__':
    num_codebooks = 4

    model = ToucanTTS(num_codebooks=num_codebooks)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    print(" TESTING TRAINING ")

    dummy_text_batch = torch.randint(low=0, high=2, size=[3, 3, 62]).float()  # [Batch, Sequence Length, Features per Phone]
    dummy_text_lens = torch.LongTensor([2, 3, 3])

    dummy_speech_batch = torch.randn([3, 9, 30, 1024])  # [Batch, Sequence Length, Spectrogram Buckets]
    dummy_speech_lens = torch.LongTensor([10, 30, 20])

    dummy_durations = torch.LongTensor([[10, 0, 0], [10, 15, 5], [5, 5, 10]])
    dummy_pitch = torch.Tensor([[[1.0], [0.], [0.]], [[1.1], [1.2], [0.8]], [[1.1], [1.2], [0.8]]])
    dummy_energy = torch.Tensor([[[1.0], [1.3], [0.]], [[1.1], [1.4], [0.8]], [[1.1], [1.2], [0.8]]])

    dummy_utterance_embed = torch.randn([3, 512])  # [Batch, Dimensions of Speaker Embedding]
    dummy_language_id = torch.LongTensor([5, 3, 2]).unsqueeze(1)

    ce, rl, mlm, dl, pl, el = model(dummy_text_batch,
                                    dummy_text_lens,
                                    dummy_speech_batch,
                                    dummy_speech_lens,
                                    dummy_durations,
                                    dummy_pitch,
                                    dummy_energy,
                                    utterance_embedding=dummy_utterance_embed,
                                    lang_ids=dummy_language_id,
                                    codebook_curriculum=num_codebooks)

    loss = ce + rl + mlm + dl + pl + el
    print(loss)
    loss.backward()

    print(" TESTING INFERENCE ")
    dummy_text_batch = torch.randint(low=0, high=2, size=[12, 62]).float()  # [Sequence Length, Features per Phone]
    dummy_utterance_embed = torch.randn([512])  # [Dimensions of Speaker Embedding]
    dummy_language_id = torch.LongTensor([2])
    print(model.inference(dummy_text_batch,
                          utterance_embedding=dummy_utterance_embed,
                          lang_id=dummy_language_id).shape)
