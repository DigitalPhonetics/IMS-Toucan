import torch

from Architectures.GeneralLayers.Conformer import Conformer
from Architectures.ToucanTTS.Glow import Glow
from Architectures.ToucanTTS.ToucanTTSLoss import ToucanTTSLoss
from Utility.utils import initialize
from Utility.utils import make_non_pad_mask


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
                 attention_dimension=384,
                 attention_heads=4,
                 positionwise_conv_kernel_size=1,
                 use_scaled_positional_encoding=True,
                 init_type="xavier_uniform",
                 use_macaron_style_in_conformer=True,
                 use_cnn_in_conformer=True,

                 # decoder
                 decoder_layers=6,
                 decoder_units=1536,
                 decoder_concat_after=False,
                 conformer_decoder_kernel_size=31,  # 31 works for spectrograms
                 decoder_normalize_before=True,
                 transformer_dec_dropout_rate=0.1,
                 transformer_dec_positional_dropout_rate=0.1,
                 transformer_dec_attn_dropout_rate=0.1,

                 # post glow
                 glow_kernel_size=5,
                 glow_blocks=12,
                 glow_layers=3,

                 # additional features
                 utt_embed_dim=192,  # 192 dim speaker embedding + 16 dim prosody embedding optionally (see older version, this one doesn't use the prosody embedding)
                 lang_embs=None,
                 lang_emb_size=None,
                 integrate_language_embedding_into_encoder_out=False,
                 embedding_integration="AdaIN",  # ["AdaIN" | "ConditionalLayerNorm" | "ConcatProject"]
                 ):
        super().__init__()

        self.config = {
            "input_feature_dimensions"                     : input_feature_dimensions,
            "attention_dimension"                          : attention_dimension,
            "attention_heads"                              : attention_heads,
            "positionwise_conv_kernel_size"                : positionwise_conv_kernel_size,
            "use_scaled_positional_encoding"               : use_scaled_positional_encoding,
            "init_type"                                    : init_type,
            "use_macaron_style_in_conformer"               : use_macaron_style_in_conformer,
            "use_cnn_in_conformer"                         : use_cnn_in_conformer,
            "decoder_layers"                               : decoder_layers,
            "decoder_units"                                : decoder_units,
            "decoder_concat_after"                         : decoder_concat_after,
            "conformer_decoder_kernel_size"                : conformer_decoder_kernel_size,
            "decoder_normalize_before"                     : decoder_normalize_before,
            "transformer_dec_dropout_rate"                 : transformer_dec_dropout_rate,
            "transformer_dec_positional_dropout_rate"      : transformer_dec_positional_dropout_rate,
            "transformer_dec_attn_dropout_rate"            : transformer_dec_attn_dropout_rate,
            "utt_embed_dim"                                : utt_embed_dim,
            "lang_embs"                                    : lang_embs,
            "lang_emb_size"                                : lang_emb_size,
            "embedding_integration"                        : embedding_integration,
            "glow_kernel_size"                             : glow_kernel_size,
            "glow_blocks"                                  : glow_blocks,
            "glow_layers"                                  : glow_layers,
            "integrate_language_embedding_into_encoder_out": integrate_language_embedding_into_encoder_out
        }

        self.input_feature_dimensions = input_feature_dimensions
        self.attention_dimension = attention_dimension
        self.use_scaled_pos_enc = use_scaled_positional_encoding
        self.multilingual_model = lang_embs is not None
        self.multispeaker_model = utt_embed_dim is not None
        self.integrate_language_embedding_into_encoder_out = integrate_language_embedding_into_encoder_out
        self.use_conditional_layernorm_embedding_integration = embedding_integration in ["AdaIN", "ConditionalLayerNorm"]

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
                                 use_output_norm=embedding_integration not in ["AdaIN", "ConditionalLayerNorm"],
                                 utt_embed=utt_embed_dim,
                                 embedding_integration=embedding_integration)

        # due to the nature of the residual vector quantization, we have to predict the codebooks in a hierarchical way.
        self.output_projection = torch.nn.Linear(attention_dimension, 128)

        self.post_flow = Glow(
            in_channels=128,
            hidden_channels=attention_dimension,  # post_glow_hidden
            kernel_size=glow_kernel_size,  # post_glow_kernel_size
            dilation_rate=1,
            n_blocks=glow_blocks,  # post_glow_n_blocks (original 12 in paper)
            n_layers=glow_layers,  # post_glow_n_block_layers (original 3 in paper)
            n_split=4,
            n_sqz=2,
            text_condition_channels=attention_dimension,
            share_cond_layers=False,  # post_share_cond_layers
            share_wn_layers=4,
            sigmoid_scale=False,
            condition_integration_projection=torch.nn.Conv1d(128 + attention_dimension, attention_dimension, 5, padding=2)
        )

        # initialize parameters
        self._reset_parameters(init_type=init_type)
        if lang_embs is not None:
            torch.nn.init.normal_(self.encoder.language_embedding.weight, mean=0, std=attention_dimension ** -0.5)

        self.criterion = ToucanTTSLoss()

    def forward(self,
                text_tensors,
                gold_speech,
                speech_lengths,
                spk_embed,
                return_feats=False,
                run_glow=True
                ):
        """
        Args:
            return_feats (Boolean): whether to return the predicted spectrogram
            text_tensors (LongTensor): Batch of padded text vectors (B, Tmax).
            gold_speech (Tensor): Batch of padded target features (B, Lmax, odim).
            speech_lengths (LongTensor): Batch of the lengths of each target (B,).
            run_glow (Bool): Whether to detach the inputs to the normalizing flow for stability.
        """
        outs, glow_loss = self._forward(text_tensors=text_tensors,
                                        gold_speech=gold_speech,
                                        speech_lengths=speech_lengths,
                                        spk_embed=spk_embed,
                                        is_inference=False,
                                        run_glow=run_glow)

        # calculate loss
        regression_loss = self.criterion(predicted_features=outs,
                                         gold_features=gold_speech,
                                         features_lengths=speech_lengths)

        if return_feats:
            return regression_loss, glow_loss, outs
        return regression_loss, glow_loss

    def _forward(self,
                 text_tensors,
                 gold_speech=None,
                 speech_lengths=None,
                 spk_embed=None,
                 is_inference=False,
                 run_glow=False):

        # decoding spectrogram
        decoder_masks = make_non_pad_mask(speech_lengths, device=speech_lengths.device).unsqueeze(-2) if speech_lengths is not None and not is_inference else None
        decoded_speech, _ = self.decoder(text_tensors, decoder_masks, utterance_embedding=spk_embed)

        preliminary_spectrogram = self.output_projection(decoded_speech)

        if is_inference:
            if run_glow:
                refined_codec_frames = self.post_flow(tgt_mels=gold_speech, infer=is_inference, mel_out=preliminary_spectrogram, encoded_texts=text_tensors, tgt_nonpadding=None)
            else:
                refined_codec_frames = preliminary_spectrogram
            return refined_codec_frames
        else:
            if run_glow:
                glow_loss = self.post_flow(tgt_mels=gold_speech, infer=is_inference, mel_out=preliminary_spectrogram, encoded_texts=text_tensors, tgt_nonpadding=decoder_masks)
            else:
                glow_loss = None
            return preliminary_spectrogram, glow_loss

    @torch.inference_mode()
    def inference(self,
                  text,
                  spk_embed,
                  run_glow=True):
        """
        Args:
            text (LongTensor): Input sequence of characters (T,).
            run_glow (bool): whether to use the output of the glow or of the out_projection to generate codec frames
        """
        self.eval()

        # setup batch axis
        ilens = torch.tensor([text.shape[0]], dtype=torch.long, device=text.device)
        text_pseudobatched, speech_pseudobatched = text.unsqueeze(0), None
        spk_embed = spk_embed.unsqueeze(0)

        outs = self._forward(text_pseudobatched,
                             ilens,
                             speech_pseudobatched,
                             spk_embed,
                             is_inference=True,
                             run_glow=run_glow)  # (1, L, odim)
        self.train()

        return outs.squeeze().transpose(0, 1)

    def _reset_parameters(self, init_type="xavier_uniform"):
        # initialize parameters
        if init_type != "pytorch":
            initialize(self, init_type)

    def reset_postnet(self, init_type="xavier_uniform"):
        # useful for after they explode
        initialize(self.post_flow, init_type)


if __name__ == '__main__':
    model = ToucanTTS()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(sum(p.numel() for p in model.post_flow.parameters() if p.requires_grad))

    print(" TESTING INFERENCE ")
    dummy_text_batch = torch.randint(low=0, high=2, size=[30, 384]).float()  # [Sequence Length, Features per Phone]
    dummy_spk_batch = torch.randint(low=0, high=2, size=[192]).float()
    print(model.inference(dummy_text_batch, dummy_spk_batch).shape)

    print(" TESTING TRAINING ")

    dummy_text_batch = torch.randint(low=0, high=2, size=[3, 30, 384]).float()  # [Batch, Sequence Length, Features per Phone]
    dummy_spk_batch = torch.randint(low=0, high=2, size=[3, 192]).float()  # [Batch, Sequence Length, Features per Phone]

    dummy_speech_batch = torch.randn([3, 30, 128])  # [Batch, Sequence Length, Spectrogram Buckets]
    dummy_speech_lens = torch.LongTensor([10, 30, 20])

    ce, fl = model(dummy_text_batch,
                   dummy_speech_batch,
                   dummy_speech_lens,
                   dummy_spk_batch)

    print(ce)
    ce.backward()
