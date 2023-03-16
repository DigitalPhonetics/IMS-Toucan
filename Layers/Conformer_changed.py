"""
Taken from ESPNet
"""

import os
import numpy as np
from typing import Optional, Tuple, Union
from torch import Tensor, nn

import torch
import torch.nn.functional as F

from Layers.Attention import RelPositionMultiHeadAttention
from Layers.Attention import RelPositionMultiHeadedAttention
from Layers.Convolution import ConvolutionModule
from Layers.Convolution_Multihead import ConvolutionModule_Multihead
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
        cnn_module_kernel (int): Kernerl size of convolution module.
        padding_idx (int): Padding idx for input_layer=embed.

    """

    def __init__(self, idim, attention_dim=256, attention_heads=4, linear_units=2048, num_blocks=6, dropout_rate=0.1, positional_dropout_rate=0.1,
                 attention_dropout_rate=0.0, input_layer="conv2d", normalize_before=True, concat_after=False, positionwise_conv_kernel_size=1,
                 macaron_style=False, use_cnn_module=False, cnn_module_kernel=31, zero_triu=False, utt_embed=None, connect_utt_emb_at_encoder_out=True,
                 spk_emb_bottleneck_size=128):
        super(Conformer, self).__init__()
        # self.encoder_pos = RelPositionalEncoding(d_model, dropout)
        #print("idim: ")
        #print(idim)
        self.accent_attn = RelPositionMultiHeadAttention(attention_dim, attention_heads, dropout=0.0)
        #self.accent_attn = RelPositionMultiHeadedAttention(attention_heads, attention_dim, dropout_rate=0.0)
        self.encoder_pos = RelPositionalEncoding(attention_dim, dropout_rate)
        #self.norm_mha = nn.LayerNorm(384)
        self.norm_mha = nn.LayerNorm(attention_dim)
        self.norm_ff_macaron = nn.LayerNorm(attention_dim)  # for the macaron style FNN module

        self.feed_forward_macaron = nn.Sequential(
            nn.Linear(attention_dim, 2048),
            Swish(),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, attention_dim),
        )

        activation = Swish()
        self.conv_subsampling_factor = 1

        if isinstance(input_layer, torch.nn.Module):
            self.embed = input_layer
            self.pos_enc = RelPositionalEncoding(attention_dim, positional_dropout_rate)
            #RelPositionalEncoding(attention_dim, positional_dropout_rate)
            #print("np.shape(self.pos_enc)")
            #print(self.pos_enc)
        elif input_layer is None:
            self.embed = None
            self.pos_enc = torch.nn.Sequential(RelPositionalEncoding(attention_dim, positional_dropout_rate))
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        self.normalize_before = normalize_before

        self.connect_utt_emb_at_encoder_out = connect_utt_emb_at_encoder_out
        if utt_embed is not None:
            self.hs_emb_projection = torch.nn.Linear(attention_dim + spk_emb_bottleneck_size, attention_dim)
            # embedding projection derived from https://arxiv.org/pdf/1705.08947.pdf
            self.embedding_projection = torch.nn.Sequential(torch.nn.Linear(utt_embed, spk_emb_bottleneck_size),
                                                            torch.nn.Softsign())

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
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
        
        # Load acccent embeddings once
        #loads the embedding from the folder of the audio with the name "{vd,at,goi,ivg,...}_emb.pt"
        
        # accent_emb is equal to what we call lang_emb with victor
        self.accent_emb = []
        for embs in next(os.walk('/nas/projects/vokquant/IMS-Toucan_lang_emb_conformer/Preprocessing/embeds/'))[2]:
            self.accent_emb.append(torch.load(os.path.join('/nas/projects/vokquant/IMS-Toucan_lang_emb_conformer/Preprocessing/embeds/', embs )))
        self.accent_emb = np.concatenate(self.accent_emb)       # created shape: (12, 1, 192)
        self.accent_emb = torch.Tensor(self.accent_emb)
        self.accent_emb = self.accent_emb.unsqueeze(0).to('cuda:0')
        print(self.accent_emb.shape)
        #self.accent_emb = np.squeeze(self.accent_emb, axis=1)   # created shape: (12, 192)

    def forward(self, xs, masks, utterance_embedding=None, lang_embs=None, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,):
        """
        Encode input sequence.
        Args:
            utterance_embedding: embedding containing lots of conditioning signals
            step: indicator for when to start updating the embedding function
            xs (torch.Tensor): Input tensor (#batch, time, idim).
            masks (torch.Tensor): Mask tensor (#batch, time).
        Returns:
            torch.Tensor: Output tensor (#batch, time, attention_dim).
            torch.Tensor: Mask tensor (#batch, time).
        """

        if self.embed is not None:
            xs = self.embed(xs) # conv2D (not the conformer)
        
        print("xs.shape self.embed")
        print(xs.shape)
        xs_temp, pos_emb = self.encoder_pos(xs)
        xs = self.encoder_pos(xs)

        # if lang_embs is not None:
        #     print("appending lang_embs with dimension 384")
        #     #print("torch.cat([lang_embs,lang_embs],dim=-1).unsqueeze(1): ")
        #     #print(torch.cat([lang_embs,lang_embs],dim=-1).unsqueeze(1))
        #     #print("1.4 * torch.cat([lang_embs,lang_embs],dim=-1).unsqueeze(1): ")
        #     #print(1.4 * torch.cat([lang_embs,lang_embs],dim=-1).unsqueeze(1))
        #     print("xs before and after torch.cat lang_emb:")
        #     print(xs.shape)
        #     xs = xs + torch.cat([lang_embs,lang_embs],dim=-1).unsqueeze(1)
        #     print(xs.shape)

        if utterance_embedding is not None and not self.connect_utt_emb_at_encoder_out:
            print("xs.shape: ")
            print(xs.shape)
            xs = self._integrate_with_utt_embed(xs, utterance_embedding)
            print(xs.shape)
        #xs_not_encoded = xs.copy
        print("before xs: ")
        #print(xs.shape)
        #xs = self.pos_enc(xs)
        #xs = self.pos_enc(xs)
        print(len(xs))
        #print(xs[0].shape)
        src_mask = torch.rand(8, 384, 63)
        
        xs, masks = self.encoders(xs, None) # src_mask = None
        if isinstance(xs, tuple):
            xs = xs[0]

        residual = xs

        if self.normalize_before:
            #xs = self.after_norm(xs)
            #xs = self.norm_mha(xs)
            xs = self.norm_ff_macaron(xs)

        self.dropout = nn.Dropout(0.1)
        xs = residual + 0.5 * self.dropout(self.feed_forward_macaron(xs))

        if not self.normalize_before:
            xs = self.norm_mha(xs)

        # multi-headed self-attention module
        residual = xs

        if self.normalize_before:
            #xs = self.after_norm(xs)
            xs = self.norm_mha(xs)

        # xs, pos_emb = self.encoder_pos(xs)
        #x, pos_emb = self.encoder_pos(xs)
        #print(x, pos_emb)
        print("self.accent_emb.repeat(1,1,1,2).squeeze(2).permute(1, 0, 2)")
        #print(self.accent_emb.repeat(1,1,1,2).squeeze(2).permute(1, 0, 2).shape) #torch.Size([12, 1, 384])
        print(self.accent_emb.repeat(1,1,1,2).squeeze(2).permute(2, 0, 1).shape) #torch.Size([384, 1, 12])
        #print(self.accent_emb.repeat(1,1,1,2).squeeze(2).shape) 
        print("pos_emb.shape: ")
        #pos_emb = xs[1]
        print("pos_emb.shape") # torch.Size([1, 125, 384]
        print(pos_emb.shape) # torch.Size([1, 125, 384]
        accent_att = self.accent_attn(
            # lang_embs,
            #lang_embs.repeat(1,2),
            # lang_embs,
            # lang_embs,
            self.accent_emb.repeat(1,1,1,2).squeeze(2).permute(1, 0, 2),
            self.accent_emb.repeat(1,1,1,2).squeeze(2).permute(1, 0, 2),
            self.accent_emb.repeat(1,1,1,2).squeeze(2).permute(1, 0, 2),
            pos_emb=pos_emb,
            attn_mask=None,
            key_padding_mask=None,
            )[0]
        x = residual + self.dropout(accent_att)

        if utterance_embedding is not None and self.connect_utt_emb_at_encoder_out:
            xs = self._integrate_with_utt_embed(xs, utterance_embedding)

        return xs, masks

    def _integrate_with_utt_embed(self, hs, utt_embeddings):
        # project embedding into smaller space
        speaker_embeddings_projected = self.embedding_projection(utt_embeddings)
        # concat hidden states with spk embeds and then apply projection
        speaker_embeddings_expanded = F.normalize(speaker_embeddings_projected).unsqueeze(1).expand(-1, hs.size(1), -1)
        hs = self.hs_emb_projection(torch.cat([hs, speaker_embeddings_expanded], dim=-1))
        return hs
