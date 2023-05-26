import torch
from torch.nn import Linear
from torch.nn import Sequential
from torch.nn import Tanh
from torch.nn import LeakyReLU
from torch.nn import LayerNorm

from Layers.Conformer import Conformer
from Layers.DurationPredictor import DurationPredictor
from Layers.LengthRegulator import LengthRegulator
from Layers.PostNet import PostNet
from Layers.VariancePredictor import VariancePredictor
from Preprocessing.articulatory_features import get_feature_to_index_lookup
from TrainingInterfaces.Text_to_Spectrogram.ToucanTTS.Glow import Glow
from TrainingInterfaces.Text_to_Spectrogram.ToucanTTS.ToucanTTSLoss import ToucanTTSLoss
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

    Things that were tried, but showed inferior performance:
    - Stochastic Duration Prediction
    - Stochastic Pitch Prediction
    - Stochastic Energy prediction
    """

    def __init__(self,
                 # network structure related
                 input_feature_dimensions=62,
                 output_spectrogram_channels=80,
                 attention_dimension=384,
                 attention_heads=4,
                 positionwise_conv_kernel_size=1,
                 use_scaled_positional_encoding=True,
                 init_type="xavier_uniform",
                 use_macaron_style_in_conformer=True,
                 use_cnn_in_conformer=True,

                 # encoder
                 encoder_layers=6,
                 encoder_units=1536,
                 encoder_normalize_before=True,
                 encoder_concat_after=False,
                 conformer_encoder_kernel_size=7,
                 transformer_enc_dropout_rate=0.2,
                 transformer_enc_positional_dropout_rate=0.2,
                 transformer_enc_attn_dropout_rate=0.2,

                 # decoder
                 decoder_layers=6,
                 decoder_units=1536,
                 decoder_concat_after=False,
                 conformer_decoder_kernel_size=31,
                 decoder_normalize_before=True,
                 transformer_dec_dropout_rate=0.2,
                 transformer_dec_positional_dropout_rate=0.2,
                 transformer_dec_attn_dropout_rate=0.2,

                 # duration predictor
                 duration_predictor_layers=3,
                 duration_predictor_chans=256,
                 duration_predictor_kernel_size=3,
                 duration_predictor_dropout_rate=0.2,

                 # pitch predictor
                 pitch_predictor_layers=7,  # 5 in espnet
                 pitch_predictor_chans=256,
                 pitch_predictor_kernel_size=5,
                 pitch_predictor_dropout=0.5,
                 pitch_embed_kernel_size=1,
                 pitch_embed_dropout=0.0,

                 # energy predictor
                 energy_predictor_layers=2,
                 energy_predictor_chans=256,
                 energy_predictor_kernel_size=3,
                 energy_predictor_dropout=0.5,
                 energy_embed_kernel_size=1,
                 energy_embed_dropout=0.0,

                 # additional features
                 utt_embed_dim=64,
                 lang_embs=8000,
                 sent_embed_dim=None,
                 sent_embed_adaptation=False,
                 sent_embed_encoder=False,
                 sent_embed_decoder=False,
                 sent_embed_each=False,
                 sent_embed_postnet=False,
                 concat_sent_style=False,
                 use_concat_projection=False,
                 use_sent_style_loss=False,
                 pre_embed=False,
                 word_embed_dim=None,
                 style_sent=False,
                 static_speaker_embed=False):
        super().__init__()

        self.input_feature_dimensions = input_feature_dimensions
        self.output_spectrogram_channels = output_spectrogram_channels
        self.attention_dimension = attention_dimension
        self.use_scaled_pos_enc = use_scaled_positional_encoding
        self.multilingual_model = lang_embs is not None
        self.multispeaker_model = utt_embed_dim is not None
        self.use_sent_embed = sent_embed_dim is not None
        self.sent_embed_adaptation = sent_embed_adaptation and self.use_sent_embed
        self.concat_sent_style = concat_sent_style and self.multispeaker_model and self.use_sent_embed
        self.use_concat_projection = use_concat_projection and self.concat_sent_style
        self.use_sent_style_loss = use_sent_style_loss and self.multispeaker_model and self.use_sent_embed
        self.sent_embed_postnet = sent_embed_postnet and self.use_sent_embed
        self.use_word_embed = word_embed_dim is not None
        self.style_sent = style_sent
        self.static_speaker_embed = static_speaker_embed

        if self.use_sent_embed:
            if self.sent_embed_adaptation:
                if self.use_sent_style_loss or self.style_sent:
                    self.sentence_embedding_adaptation = Sequential(Linear(sent_embed_dim, sent_embed_dim // 2),
                                                                    Tanh(),
                                                                    Linear(sent_embed_dim // 2, sent_embed_dim // 4),
                                                                    Tanh(),
                                                                    Linear(sent_embed_dim // 4, 64))
                    sent_embed_dim = 64
                else:
                    self.sentence_embedding_adaptation = Sequential(Linear(sent_embed_dim, sent_embed_dim // 2),
                                                                    Tanh(),
                                                                    Linear(sent_embed_dim // 2, sent_embed_dim // 4),
                                                                    Tanh(),
                                                                    Linear(sent_embed_dim // 4, 64))
                    sent_embed_dim = 64
            '''
            self.utt_embed_bottleneck = Sequential(Linear(512, 256), 
                                                    Tanh(), 
                                                    Linear(256, 128),
                                                    Tanh(),
                                                    Linear(128, 64),
                                                    Tanh(),
                                                    Linear(64, 32),
                                                    Tanh(),
                                                    Linear(32, 16),
                                                    Tanh(),
                                                    Linear(16, 8),
                                                    Tanh(),
                                                    Linear(8, 4))
            #utt_embed_dim = 4 # hard bottleneck, hopefully only comntains speaker timbre
            '''
            if self.static_speaker_embed:
                self.speaker_embedding = torch.nn.Embedding(10, 16)
                utt_embed_dim = 16
            if self.concat_sent_style:
                if not self.static_speaker_embed:
                    self.utt_embed_bottleneck = Sequential(Linear(512, 256), 
                                                            Tanh(), 
                                                            Linear(256, 128),
                                                            Tanh(),
                                                            Linear(128, 64),
                                                            Tanh(),
                                                            Linear(64, 32),
                                                            Tanh(),
                                                            Linear(32, 16),
                                                            Tanh(),
                                                            Linear(16, 8))
                    utt_embed_dim = 8
                else:
                    self.utt_embed_bottleneck = None
                if self.use_concat_projection:
                    self.style_embedding_projection = Sequential(Linear(utt_embed_dim + sent_embed_dim, 64),
                                                                 LayerNorm(64))
                    utt_embed_dim = 64
                else:
                    utt_embed_dim = utt_embed_dim + sent_embed_dim

            if pre_embed:
                input_feature_dimensions = 62 + sent_embed_dim

        articulatory_feature_embedding = Sequential(Linear(input_feature_dimensions, 512 if pre_embed else 100), Tanh(), Linear(512 if pre_embed else 100, attention_dimension))
        self.encoder = Conformer(idim=input_feature_dimensions,
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
                                 sent_embed_dim=sent_embed_dim if sent_embed_encoder else None,
                                 sent_embed_each=sent_embed_each,
                                 pre_embed=pre_embed,
                                 word_embed_dim=word_embed_dim,
                                 use_output_norm=True)

        self.duration_predictor = DurationPredictor(idim=attention_dimension, n_layers=duration_predictor_layers,
                                                    n_chans=duration_predictor_chans,
                                                    kernel_size=duration_predictor_kernel_size,
                                                    dropout_rate=duration_predictor_dropout_rate,
                                                    utt_embed_dim=utt_embed_dim)

        self.pitch_predictor = VariancePredictor(idim=attention_dimension, n_layers=pitch_predictor_layers,
                                                 n_chans=pitch_predictor_chans,
                                                 kernel_size=pitch_predictor_kernel_size,
                                                 dropout_rate=pitch_predictor_dropout,
                                                 utt_embed_dim=utt_embed_dim)

        self.energy_predictor = VariancePredictor(idim=attention_dimension, n_layers=energy_predictor_layers,
                                                  n_chans=energy_predictor_chans,
                                                  kernel_size=energy_predictor_kernel_size,
                                                  dropout_rate=energy_predictor_dropout,
                                                  utt_embed_dim=utt_embed_dim)

        self.pitch_embed = Sequential(torch.nn.Conv1d(in_channels=1,
                                                      out_channels=attention_dimension,
                                                      kernel_size=pitch_embed_kernel_size,
                                                      padding=(pitch_embed_kernel_size - 1) // 2),
                                      torch.nn.Dropout(pitch_embed_dropout))

        self.energy_embed = Sequential(torch.nn.Conv1d(in_channels=1, out_channels=attention_dimension, kernel_size=energy_embed_kernel_size,
                                                       padding=(energy_embed_kernel_size - 1) // 2),
                                       torch.nn.Dropout(energy_embed_dropout))

        self.length_regulator = LengthRegulator()

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
                                 utt_embed=None,
                                 sent_embed_dim=sent_embed_dim if sent_embed_decoder else None,
                                 sent_embed_each=sent_embed_each,
                                 use_output_norm=False)

        self.feat_out = Linear(attention_dimension, output_spectrogram_channels)

        if self.sent_embed_postnet:
            self.decoder_out_sent_emb_projection = Sequential(Linear(output_spectrogram_channels + sent_embed_dim,
                                                                    output_spectrogram_channels),
                                                            LayerNorm(output_spectrogram_channels))

        self.conv_postnet = PostNet(idim=0,
                                    odim=output_spectrogram_channels,
                                    n_layers=5,
                                    n_chans=256,
                                    n_filts=5,
                                    use_batch_norm=True,
                                    dropout_rate=0.5)

        self.post_flow = Glow(
            in_channels=output_spectrogram_channels,
            hidden_channels=192,  # post_glow_hidden
            kernel_size=5,  # post_glow_kernel_size
            dilation_rate=1,
            n_blocks=18,  # post_glow_n_blocks (original 12 in paper)
            n_layers=4,  # post_glow_n_block_layers (original 3 in paper)
            n_split=4,
            n_sqz=2,
            text_condition_channels=attention_dimension,
            share_cond_layers=False,  # post_share_cond_layers
            share_wn_layers=4,
            sigmoid_scale=False,
            condition_integration_projection=torch.nn.Conv1d(output_spectrogram_channels + attention_dimension, attention_dimension, 5, padding=2)
        )

        # initialize parameters
        self._reset_parameters(init_type=init_type)
        if lang_embs is not None:
            torch.nn.init.normal_(self.encoder.language_embedding.weight, mean=0, std=attention_dimension ** -0.5)

        self.criterion = ToucanTTSLoss()
        if self.use_sent_style_loss:
            print("Using sentence style loss")
            self.mse_criterion = torch.nn.MSELoss(reduction='mean')

    def forward(self,
                text_tensors,
                text_lengths,
                gold_speech,
                speech_lengths,
                gold_durations,
                gold_pitch,
                gold_energy,
                utterance_embedding,
                speaker_id=None,
                sentence_embedding=None,
                word_embedding=None,
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
        predicted_durations, \
        predicted_pitch, \
        predicted_energy, \
        glow_loss, \
        sent_style_loss = self._forward(text_tensors=text_tensors,
                                  text_lengths=text_lengths,
                                  gold_speech=gold_speech,
                                  speech_lengths=speech_lengths,
                                  gold_durations=gold_durations,
                                  gold_pitch=gold_pitch,
                                  gold_energy=gold_energy,
                                  utterance_embedding=utterance_embedding,
                                  speaker_id=speaker_id,
                                  sentence_embedding=sentence_embedding,
                                  word_embedding=word_embedding,
                                  is_inference=False,
                                  lang_ids=lang_ids,
                                  run_glow=run_glow)

        # calculate loss
        l1_loss, duration_loss, pitch_loss, energy_loss = self.criterion(after_outs=after_outs,
                                                                         # if a regular PostNet is used, the post-PostNet outs have to go here. The flow has its own loss though, so we hard-code this to None
                                                                         before_outs=before_outs,
                                                                         gold_spectrograms=gold_speech,
                                                                         spectrogram_lengths=speech_lengths,
                                                                         text_lengths=text_lengths,
                                                                         gold_durations=gold_durations,
                                                                         predicted_durations=predicted_durations,
                                                                         predicted_pitch=predicted_pitch,
                                                                         predicted_energy=predicted_energy,
                                                                         gold_pitch=gold_pitch,
                                                                         gold_energy=gold_energy)

        if return_mels:
            if after_outs is None:
                after_outs = before_outs
            return l1_loss, duration_loss, pitch_loss, energy_loss, glow_loss, sent_style_loss, after_outs,
        return l1_loss, duration_loss, pitch_loss, energy_loss, glow_loss, sent_style_loss

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
                 speaker_id=None,
                 sentence_embedding=None,
                 word_embedding=None,
                 lang_ids=None,
                 run_glow=True):

        if not self.multilingual_model:
            lang_ids = None

        if self.use_sent_style_loss:
            if utterance_embedding is None:
                raise TypeError("utterance embedding is None")
            sentence_embedding_gold = utterance_embedding

        if not self.multispeaker_model:
            utterance_embedding = None
        else:
            if self.static_speaker_embed:
                utterance_embedding = self.speaker_embedding(speaker_id)
            else:
                utterance_embedding = torch.nn.functional.normalize(utterance_embedding)

        if not self.use_sent_embed:
            sentence_embedding = None
            utterance_embedding_decoder = None
        else:
            sentence_embedding = torch.nn.functional.normalize(sentence_embedding)
            if self.sent_embed_adaptation:
                # forward sentence embedding adaptation
                sentence_embedding = self.sentence_embedding_adaptation(sentence_embedding)
            #utterance_embedding = sentence_embedding if self.style_sent and is_inference else utterance_embedding
            #utterance_embedding_decoder = self.utt_embed_bottleneck(utterance_embedding)
            utterance_embedding_decoder = None
            utterance_embedding = sentence_embedding if self.style_sent else utterance_embedding

        if not self.use_word_embed:
            word_embedding = None
            word_boundaries_batch = None
        else:
            # get word boundaries
            word_boundaries_batch = []
            for batch_id, batch in enumerate(text_tensors):
                word_boundaries = []
                for phoneme_index, phoneme_vector in enumerate(batch):
                    if phoneme_vector[get_feature_to_index_lookup()["word-boundary"]] == 1:
                        word_boundaries.append(phoneme_index)
                word_boundaries.append(text_lengths[batch_id].cpu().numpy()-1) # marker for last word of sentence
                word_boundaries_batch.append(torch.tensor(word_boundaries))

        if self.concat_sent_style:
            if self.utt_embed_bottleneck is not None:
                utterance_embedding = self.utt_embed_bottleneck(utterance_embedding)
            if self.use_sent_style_loss:
                utterance_embedding = _concat_sent_utt(utt_embeddings=utterance_embedding, 
                                                   sent_embeddings=sentence_embedding, 
                                                   projection=self.style_embedding_projection if self.use_concat_projection else None)
            else:
                utterance_embedding = _concat_sent_utt(utt_embeddings=utterance_embedding, 
                                                        sent_embeddings=sentence_embedding, 
                                                        projection=self.style_embedding_projection if self.use_concat_projection else None)

        # encoding the texts
        text_masks = make_non_pad_mask(text_lengths, device=text_lengths.device).unsqueeze(-2)
        padding_masks = make_pad_mask(text_lengths, device=text_lengths.device)
        if self.use_sent_style_loss:
            encoded_texts, _ = self.encoder(text_tensors,
                                            text_masks,
                                            utterance_embedding=utterance_embedding,
                                            sentence_embedding=sentence_embedding,
                                            word_embedding=word_embedding,
                                            word_boundaries=word_boundaries_batch,
                                            lang_ids=lang_ids)  # (B, Tmax, adim)
        else:
            encoded_texts, _ = self.encoder(text_tensors,
                                            text_masks,
                                            utterance_embedding=utterance_embedding,
                                            sentence_embedding=sentence_embedding,
                                            word_embedding=word_embedding,
                                            word_boundaries=word_boundaries_batch,
                                            lang_ids=lang_ids)  # (B, Tmax, adim)

        if is_inference:
            # predicting pitch, energy and durations
            pitch_predictions = self.pitch_predictor(encoded_texts, padding_mask=None, utt_embed=utterance_embedding)
            energy_predictions = self.energy_predictor(encoded_texts, padding_mask=None, utt_embed=utterance_embedding)
            predicted_durations = self.duration_predictor.inference(encoded_texts, padding_mask=None, utt_embed=utterance_embedding)

            # modifying the predictions with linguistic knowledge
            for phoneme_index, phoneme_vector in enumerate(text_tensors.squeeze(0)):
                if phoneme_vector[get_feature_to_index_lookup()["voiced"]] == 0:
                    pitch_predictions[0][phoneme_index] = 0.0
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
            pitch_predictions = self.pitch_predictor(encoded_texts.detach(), padding_mask=padding_masks.unsqueeze(-1), utt_embed=utterance_embedding)
            energy_predictions = self.energy_predictor(encoded_texts, padding_mask=padding_masks.unsqueeze(-1), utt_embed=utterance_embedding)
            predicted_durations = self.duration_predictor(encoded_texts, padding_mask=padding_masks, utt_embed=utterance_embedding)

            embedded_pitch_curve = self.pitch_embed(gold_pitch.transpose(1, 2)).transpose(1, 2)
            embedded_energy_curve = self.energy_embed(gold_energy.transpose(1, 2)).transpose(1, 2)
            enriched_encoded_texts = encoded_texts + embedded_energy_curve + embedded_pitch_curve

            upsampled_enriched_encoded_texts = self.length_regulator(enriched_encoded_texts, gold_durations)

        # decoding spectrogram
        decoder_masks = make_non_pad_mask(speech_lengths, device=speech_lengths.device).unsqueeze(-2) if speech_lengths is not None and not is_inference else None
        if self.use_sent_style_loss:
            decoded_speech, _ = self.decoder(upsampled_enriched_encoded_texts, 
                                            decoder_masks,
                                            utterance_embedding=utterance_embedding_decoder,
                                            sentence_embedding=sentence_embedding)
        else:
            decoded_speech, _ = self.decoder(upsampled_enriched_encoded_texts, 
                                            decoder_masks,
                                            utterance_embedding=utterance_embedding_decoder,
                                            sentence_embedding=sentence_embedding)
        decoded_spectrogram = self.feat_out(decoded_speech).view(decoded_speech.size(0), -1, self.output_spectrogram_channels)

        refined_spectrogram = decoded_spectrogram + self.conv_postnet(decoded_spectrogram.transpose(1, 2)).transpose(1, 2)

        # refine spectrogram further with a normalizing flow (requires warmup, so it's not always on)
        glow_loss = None
        if run_glow:
            if self.sent_embed_postnet:
                if self.use_sent_style_loss:
                    refined_spectrogram = _integrate_with_sent_embed(hs=refined_spectrogram,
                                                                    sent_embeddings=sentence_embedding,
                                                                    projection=self.decoder_out_sent_emb_projection)
                else:
                    refined_spectrogram = _integrate_with_sent_embed(hs=refined_spectrogram,
                                                                    sent_embeddings=sentence_embedding,
                                                                    projection=self.decoder_out_sent_emb_projection)
            if is_inference:
                refined_spectrogram = self.post_flow(tgt_mels=None,
                                                     infer=is_inference,
                                                     mel_out=refined_spectrogram,
                                                     encoded_texts=upsampled_enriched_encoded_texts,
                                                     tgt_nonpadding=None).squeeze()
            else:
                glow_loss = self.post_flow(tgt_mels=gold_speech,
                                           infer=is_inference,
                                           mel_out=refined_spectrogram.detach().clone(),
                                           encoded_texts=upsampled_enriched_encoded_texts.detach().clone(),
                                           tgt_nonpadding=decoder_masks)
                
        if self.use_sent_style_loss:
            sent_style_loss = self.mse_criterion(sentence_embedding, sentence_embedding_gold)
        else:
            sent_style_loss = None
        
        if is_inference:
            return decoded_spectrogram.squeeze(), \
                   refined_spectrogram.squeeze(), \
                   predicted_durations.squeeze(), \
                   pitch_predictions.squeeze(), \
                   energy_predictions.squeeze()
        else:
            return decoded_spectrogram, \
                   refined_spectrogram, \
                   predicted_durations, \
                   pitch_predictions, \
                   energy_predictions, \
                   glow_loss, \
                   sent_style_loss

    @torch.inference_mode()
    def inference(self,
                  text,
                  speech=None,
                  utterance_embedding=None,
                  speaker_id=None,
                  sentence_embedding=None,
                  word_embedding=None,
                  return_duration_pitch_energy=False,
                  lang_id=None,
                  run_postflow=True):
        """
        Args:
            text (LongTensor): Input sequence of characters (T,).
            speech (Tensor, optional): Feature sequence to extract style (N, idim).
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
        utterance_embeddings = utterance_embedding.unsqueeze(0).to(x.device) if utterance_embedding is not None else None
        sentence_embeddings = sentence_embedding.unsqueeze(0).to(x.device) if sentence_embedding is not None else None
        word_embeddings = word_embedding.unsqueeze(0).to(x.device) if word_embedding is not None else None
        speaker_id = speaker_id.to(x.device) if speaker_id is not None else None

        before_outs, \
        after_outs, \
        duration_predictions, \
        pitch_predictions, \
        energy_predictions = self._forward(xs,
                                           ilens,
                                           ys,
                                           is_inference=True,
                                           utterance_embedding=utterance_embeddings,
                                           speaker_id=speaker_id,
                                           sentence_embedding=sentence_embeddings,
                                           word_embedding=word_embeddings,
                                           lang_ids=lang_id,
                                           run_glow=run_postflow)  # (1, L, odim)
        self.train()
        if after_outs is None:
            after_outs = before_outs
        if return_duration_pitch_energy:
            return before_outs, after_outs, duration_predictions, pitch_predictions, energy_predictions
        return after_outs

    def _reset_parameters(self, init_type):
        # initialize parameters
        if init_type != "pytorch":
            initialize(self, init_type)

def _concat_sent_utt(utt_embeddings, sent_embeddings, projection):
    if projection is not None:
        utt_embeddings_enriched = projection(torch.cat([utt_embeddings, sent_embeddings], dim=1))
    else:
        utt_embeddings_enriched = torch.cat([utt_embeddings, sent_embeddings], dim=1)
    return utt_embeddings_enriched

def _integrate_with_sent_embed(hs, sent_embeddings, projection):
    # concat hidden states with sent embeds and then apply projection
    embeddings_expanded = sent_embeddings.unsqueeze(1).expand(-1, hs.size(1), -1)
    hs = projection(torch.cat([hs, embeddings_expanded], dim=-1))
    return hs


if __name__ == '__main__':
    print(sum(p.numel() for p in ToucanTTS().parameters() if p.requires_grad))

    print(" TESTING TRAINING ")

    print(" batchsize 3 ")
    dummy_text_batch = torch.randint(low=0, high=2, size=[3, 3, 62]).float()  # [Batch, Sequence Length, Features per Phone]
    dummy_text_lens = torch.LongTensor([2, 3, 3])

    dummy_speech_batch = torch.randn([3, 30, 80])  # [Batch, Sequence Length, Spectrogram Buckets]
    dummy_speech_lens = torch.LongTensor([10, 30, 20])

    dummy_durations = torch.LongTensor([[10, 0, 0], [10, 15, 5], [5, 5, 10]])
    dummy_pitch = torch.Tensor([[[1.0], [0.], [0.]], [[1.1], [1.2], [0.8]], [[1.1], [1.2], [0.8]]])
    dummy_energy = torch.Tensor([[[1.0], [1.3], [0.]], [[1.1], [1.4], [0.8]], [[1.1], [1.2], [0.8]]])

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

    loss = l1 + gl + dl + pl + el
    print(loss)
    loss.backward()

    # from Utility.utils import plot_grad_flow

    # plot_grad_flow(model.encoder.named_parameters())
    # plot_grad_flow(model.decoder.named_parameters())
    # plot_grad_flow(model.pitch_predictor.named_parameters())
    # plot_grad_flow(model.duration_predictor.named_parameters())
    # plot_grad_flow(model.post_flow.named_parameters())

    print(" batchsize 2 ")
    dummy_text_batch = torch.randint(low=0, high=2, size=[2, 3, 62]).float()  # [Batch, Sequence Length, Features per Phone]
    dummy_text_lens = torch.LongTensor([2, 3])

    dummy_speech_batch = torch.randn([2, 30, 80])  # [Batch, Sequence Length, Spectrogram Buckets]
    dummy_speech_lens = torch.LongTensor([10, 30])

    dummy_durations = torch.LongTensor([[10, 0, 0], [10, 15, 5]])
    dummy_pitch = torch.Tensor([[[1.0], [0.], [0.]], [[1.1], [1.2], [0.8]]])
    dummy_energy = torch.Tensor([[[1.0], [1.3], [0.]], [[1.1], [1.4], [0.8]]])

    dummy_utterance_embed = torch.randn([2, 64])  # [Batch, Dimensions of Speaker Embedding]
    dummy_language_id = torch.LongTensor([5, 3]).unsqueeze(1)

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
