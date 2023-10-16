"""
Taken from ESPNet, but heavily modified
"""

import torch

from Layers.Attention import RelPositionMultiHeadedAttention
from Layers.ConditionalLayerNorm import AdaIN1d
from Layers.Convolution import ConvolutionModule
from Layers.EncoderLayer import EncoderLayer
from Layers.LayerNorm import LayerNorm
from Layers.MultiLayeredConv1d import MultiLayeredConv1d
from Layers.MultiSequential import repeat
from Layers.PositionalEncoding import RelPositionalEncoding
from Layers.Swish import Swish
from Utility.utils import integrate_with_utt_embed


class Conformer(torch.nn.Module):
    """
    Conformer encoder module.

    Args:
        idim (int): Input dimension.
        attention_dim (int): Dimension of attention.
        attention_heads (int): The number of heads of multi head attention.
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        attention_dropout_rate (float): Dropout rate in attention.
        input_layer (Union[str, torch.nn.Module]): Input layer type.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
        positionwise_conv_kernel_size (int): Kernel size of positionwise conv1d layer.
        macaron_style (bool): Whether to use macaron style for positionwise layer.
        pos_enc_layer_type (str): Conformer positional encoding layer type.
        selfattention_layer_type (str): Conformer attention layer type.
        activation_type (str): Conformer activation function type.
        use_cnn_module (bool): Whether to use convolution module.
        cnn_module_kernel (int): Kernel size of convolution module.
        padding_idx (int): Padding idx for input_layer=embed.

    """

    def __init__(self, conformer_type, attention_dim=256, attention_heads=4, linear_units=2048, num_blocks=6, dropout_rate=0.1, positional_dropout_rate=0.1,
                 attention_dropout_rate=0.0, input_layer="conv2d", normalize_before=True, concat_after=False, positionwise_conv_kernel_size=1,
                 macaron_style=False, use_cnn_module=False, cnn_module_kernel=31, zero_triu=False, utt_embed=None, lang_embs=None, use_output_norm=True, use_conditional_layernorm_embedding_integration=False):
        super(Conformer, self).__init__()

        activation = Swish()
        self.conv_subsampling_factor = 1
        self.use_output_norm = use_output_norm

        if isinstance(input_layer, torch.nn.Module):
            self.embed = input_layer
            self.pos_enc = RelPositionalEncoding(attention_dim, positional_dropout_rate)
        elif input_layer is None:
            self.embed = None
            self.pos_enc = torch.nn.Sequential(RelPositionalEncoding(attention_dim, positional_dropout_rate))
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        if self.use_output_norm:
            self.output_norm = LayerNorm(attention_dim)
        self.utt_embed = utt_embed
        self.conformer_type = conformer_type
        self.use_conditional_layernorm_embedding_integration = use_conditional_layernorm_embedding_integration
        if utt_embed is not None:
            if conformer_type == "encoder":  # the encoder gets an additional conditioning signal added to its output
                if use_conditional_layernorm_embedding_integration:
                    self.encoder_embedding_projection = AdaIN1d(style_dim=utt_embed, num_features=attention_dim)
                else:
                    self.encoder_embedding_projection = torch.nn.Linear(attention_dim + utt_embed, attention_dim)
            else:
                if use_conditional_layernorm_embedding_integration:
                    self.decoder_embedding_projections = repeat(num_blocks, lambda lnum: AdaIN1d(style_dim=utt_embed, num_features=attention_dim))
                else:
                    self.decoder_embedding_projections = repeat(num_blocks, lambda lnum: torch.nn.Linear(attention_dim + utt_embed, attention_dim))
        if lang_embs is not None:
            self.language_embedding = torch.nn.Embedding(num_embeddings=lang_embs, embedding_dim=attention_dim)

        # self-attention module definition
        encoder_selfattn_layer = RelPositionMultiHeadedAttention
        encoder_selfattn_layer_args = (attention_heads, attention_dim, attention_dropout_rate, zero_triu)

        # feed-forward module definition
        positionwise_layer = MultiLayeredConv1d
        positionwise_layer_args = (attention_dim, linear_units, positionwise_conv_kernel_size, dropout_rate,)

        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (attention_dim, cnn_module_kernel, activation)

        self.encoders = repeat(num_blocks, lambda lnum: EncoderLayer(attention_dim, encoder_selfattn_layer(*encoder_selfattn_layer_args),
                                                                     positionwise_layer(*positionwise_layer_args),
                                                                     positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                                                                     convolution_layer(*convolution_layer_args) if use_cnn_module else None, dropout_rate,
                                                                     normalize_before, concat_after))

    def forward(self,
                xs,
                masks,
                utterance_embedding=None,
                lang_ids=None):
        """
        Encode input sequence.
        Args:
            utterance_embedding: embedding containing lots of conditioning signals
            lang_ids: ids of the languages per sample in the batch
            xs (torch.Tensor): Input tensor (#batch, time, idim).
            masks (torch.Tensor): Mask tensor (#batch, time).
        Returns:
            torch.Tensor: Output tensor (#batch, time, attention_dim).
            torch.Tensor: Mask tensor (#batch, time).
        """

        if self.embed is not None:
            xs = self.embed(xs)

        if lang_ids is not None:
            lang_embs = self.language_embedding(lang_ids)
            xs = xs + lang_embs  # offset phoneme representation by language specific offset

        xs = self.pos_enc(xs)

        for encoder_index, encoder in enumerate(self.encoders):
            if self.utt_embed:
                if isinstance(xs, tuple):
                    x, pos_emb = xs[0], xs[1]
                    if self.conformer_type != "encoder":
                        x = integrate_with_utt_embed(hs=x, utt_embeddings=utterance_embedding, projection=self.decoder_embedding_projections[encoder_index], embedding_training=self.use_conditional_layernorm_embedding_integration)
                    xs = (x, pos_emb)
                else:
                    xs = integrate_with_utt_embed(hs=xs, utt_embeddings=utterance_embedding, projection=self.decoder_embedding_projections[encoder_index], embedding_training=self.use_conditional_layernorm_embedding_integration)
            xs, masks = encoder(xs, masks)

        if isinstance(xs, tuple):
            xs = xs[0]

        if self.use_output_norm and not (self.utt_embed and self.conformer_type == "encoder"):
            xs = self.output_norm(xs)

        if self.utt_embed and self.conformer_type == "encoder":
            xs = integrate_with_utt_embed(hs=xs, utt_embeddings=utterance_embedding,
                                          projection=self.encoder_embedding_projection, embedding_training=self.use_conditional_layernorm_embedding_integration)

        return xs, masks
