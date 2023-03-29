"""
Taken from ESPNet
"""

import torch
from torch.nn.utils.rnn import pad_sequence

from Layers.Attention import RelPositionMultiHeadedAttention
from Layers.Convolution import ConvolutionModule
from Layers.EncoderLayer import EncoderLayer
from Layers.LayerNorm import LayerNorm
from Layers.MultiLayeredConv1d import MultiLayeredConv1d
from Layers.MultiSequential import repeat
from Layers.PositionalEncoding import RelPositionalEncoding
from Layers.Swish import Swish


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

    def __init__(self, idim, attention_dim=256, attention_heads=4, linear_units=2048, num_blocks=6, dropout_rate=0.1, positional_dropout_rate=0.1,
                 attention_dropout_rate=0.0, input_layer="conv2d", normalize_before=True, concat_after=False, positionwise_conv_kernel_size=1,
                 macaron_style=False, use_cnn_module=False, cnn_module_kernel=31, zero_triu=False, utt_embed=None, word_embed=None, lang_embs=None, use_output_norm=True):
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
        if utt_embed is not None:
            self.hs_emb_projection = torch.nn.Linear(attention_dim + utt_embed, attention_dim)
        self.word_embed = word_embed
        if word_embed is not None:
            self.word_phoneme_projection = torch.nn.Linear(attention_dim + word_embed, attention_dim)
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
                word_embedding=None,
                word_boundaries=None,
                text_lengths=None,
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

        # concat word embs
        # NOTE there is probably a more elegant and more efficient way, maybe by using pad masks
        if self.word_embed:
            xs_enhanced = []
            for batch_id, wbs in enumerate(word_boundaries):
                w_start = 0
                phoneme_sequence = []
                for i, wb_id in enumerate(wbs):
                    # get phoneme embeddings corresponding to words according to word boundaries
                    if i+1 == len(wbs):
                        phoneme_embeds = xs[batch_id, w_start:text_lengths[batch_id]]
                    else:
                        phoneme_embeds = xs[batch_id, w_start:wb_id+1]
                    # get cooresponding word embedding
                    try:
                        word_embed = word_embedding[batch_id, i]
                    except IndexError:
                        # seems like in some cases (just very few now) the length of word embeddings doesn't align with the one of word boundaries
                        # in this case we just take the last word embedding again for the last phonemes
                        word_embed = word_embedding[batch_id, -1]
                    # concatenate phoneme embeddings with word embedding
                    phoneme_embeds = self._cat_with_word_embed(phoneme_embeddings=phoneme_embeds, word_embedding=word_embed)
                    phoneme_sequence.append(phoneme_embeds)
                    w_start = wb_id+1
                # put whole phoneme sequence back together
                phoneme_sequence = torch.cat(phoneme_sequence, dim=0)
                xs_enhanced.append(phoneme_sequence)
            # pad phoneme sequences to get whole batch
            xs_enhanced_padded = pad_sequence(xs_enhanced, batch_first=True)
            # apply projection
            xs = self.word_phoneme_projection(xs_enhanced_padded)


        if lang_ids is not None:
            lang_embs = self.language_embedding(lang_ids)
            xs = xs + lang_embs  # offset phoneme representation by language specific offset

        xs = self.pos_enc(xs)

        xs, masks = self.encoders(xs, masks)
        if isinstance(xs, tuple):
            xs = xs[0]

        if self.use_output_norm:
            xs = self.output_norm(xs)

        if self.utt_embed:
            xs = self._integrate_with_utt_embed(hs=xs, utt_embeddings=utterance_embedding)

        return xs, masks

    def _integrate_with_utt_embed(self, hs, utt_embeddings):
        # concat hidden states with spk embeds and then apply projection
        embeddings_expanded = torch.nn.functional.normalize(utt_embeddings).unsqueeze(1).expand(-1, hs.size(1), -1)
        hs = self.hs_emb_projection(torch.cat([hs, embeddings_expanded], dim=-1))
        return hs
    
    def _cat_with_word_embed(self, phoneme_embeddings, word_embedding):
        # concat phoneme embeddings with corresponding word embedding and then apply projection
        word_embeddings_expanded = torch.nn.functional.normalize(word_embedding, dim=0).unsqueeze(0).expand(phoneme_embeddings.size(0), -1)
        phoneme_embeddings = torch.cat([phoneme_embeddings, word_embeddings_expanded], dim=-1)
        return phoneme_embeddings
