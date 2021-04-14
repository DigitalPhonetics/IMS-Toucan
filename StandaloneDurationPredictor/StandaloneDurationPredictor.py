"""
Taken from ESPNet
"""

import torch

from StandaloneDurationPredictor.Conformer import Conformer
from StandaloneDurationPredictor.DurationPredictor import DurationPredictor
from StandaloneDurationPredictor.utils import make_non_pad_mask
from StandaloneDurationPredictor.utils import make_pad_mask


class StandaloneDurationPredictor(torch.nn.Module):

    def __init__(self, path_to_model=None, number_of_phonemes=133, adim=384, aheads=4, elayers=6, eunits=1536, positionwise_conv_kernel_size=1, encoder_normalize_before=True,
                 encoder_concat_after=False, use_macaron_style_in_conformer=True, use_cnn_in_conformer=True, conformer_enc_kernel_size=7, duration_predictor_layers=2,
                 duration_predictor_chans=256, duration_predictor_kernel_size=3, transformer_enc_dropout_rate=0.2, transformer_enc_positional_dropout_rate=0.2,
                 transformer_enc_attn_dropout_rate=0.2, duration_predictor_dropout_rate=0.2):
        super().__init__()
        self.idim = number_of_phonemes
        self.eos = 1
        self.padding_idx = 0
        encoder_input_layer = torch.nn.Embedding(num_embeddings=number_of_phonemes, embedding_dim=adim, padding_idx=self.padding_idx)
        self.encoder = Conformer(idim=number_of_phonemes, attention_dim=adim, attention_heads=aheads, linear_units=eunits, num_blocks=elayers,
                                 input_layer=encoder_input_layer, dropout_rate=transformer_enc_dropout_rate,
                                 positional_dropout_rate=transformer_enc_positional_dropout_rate, attention_dropout_rate=transformer_enc_attn_dropout_rate,
                                 normalize_before=encoder_normalize_before, concat_after=encoder_concat_after,
                                 positionwise_conv_kernel_size=positionwise_conv_kernel_size, macaron_style=use_macaron_style_in_conformer,
                                 use_cnn_module=use_cnn_in_conformer, cnn_module_kernel=conformer_enc_kernel_size, zero_triu=False)
        self.duration_predictor = DurationPredictor(idim=adim, n_layers=duration_predictor_layers, n_chans=duration_predictor_chans,
                                                    kernel_size=duration_predictor_kernel_size, dropout_rate=duration_predictor_dropout_rate)
        if path_to_model is not None:
            self.load_state_dict(torch.load(path_to_model, map_location='cpu')["dp"])

    def forward(self, text):
        self.eval()
        text_lens = torch.tensor([text.shape[0]], dtype=torch.long, device=text.device)
        text_masks = make_non_pad_mask(text_lens).to(text_lens.device).unsqueeze(-2)
        encoded_texts, _ = self.encoder(text.unsqueeze(0), text_masks)
        d_masks = make_pad_mask(text_lens).to(text.unsqueeze(0).device)
        d_outs = self.duration_predictor.inference(encoded_texts, d_masks)
        return d_outs[0]
