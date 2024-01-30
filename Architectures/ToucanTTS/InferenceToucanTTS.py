import dotwiz
import torch

from Architectures.GeneralLayers.Conformer import Conformer
from Architectures.ToucanTTS.Glow import Glow


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
        decoder_layers = config.decoder_layers
        decoder_units = config.decoder_units
        decoder_concat_after = config.decoder_concat_after
        conformer_decoder_kernel_size = config.conformer_decoder_kernel_size
        decoder_normalize_before = config.decoder_normalize_before
        transformer_dec_dropout_rate = config.transformer_dec_dropout_rate
        transformer_dec_positional_dropout_rate = config.transformer_dec_positional_dropout_rate
        transformer_dec_attn_dropout_rate = config.transformer_dec_attn_dropout_rate
        utt_embed_dim = config.utt_embed_dim
        lang_embs = config.lang_embs
        embedding_integration = config.embedding_integration
        glow_kernel_size = config.glow_kernel_size
        glow_blocks = config.glow_blocks
        glow_layers = config.glow_layers
        integrate_language_embedding_into_encoder_out = config.integrate_language_embedding_into_encoder_out

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

        self.load_state_dict(weights)
        self.eval()

    def _forward(self,
                 text_tensors,
                 glow_sampling_temperature=0.2):
        # decoding spectrogram
        decoded_speech, _ = self.decoder(text_tensors, None, utterance_embedding=None)
        frames = self.output_projection(decoded_speech)
        refined_codec_frames = self.post_flow(tgt_mels=None, infer=True, mel_out=frames, encoded_texts=text_tensors, tgt_nonpadding=None, glow_sampling_temperature=glow_sampling_temperature)
        return refined_codec_frames

    @torch.inference_mode()
    def forward(self,
                text,
                glow_sampling_temperature=0.2):
        """
        Generate the sequence of spectrogram frames given the sequence of vectorized phonemes.

        Args:
            text: input sequence of vectorized phonemes

        Returns:
            features spectrogram

        """
        outs = self._forward(text.unsqueeze(0),
                             glow_sampling_temperature=glow_sampling_temperature)
        return outs.squeeze().transpose(0, 1)

    def store_inverse_all(self):
        def remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(remove_weight_norm)
