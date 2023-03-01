import torch
from torch import nn

from TrainingInterfaces.Text_to_Spectrogram.PortaSpeech.PortaSpeechLayers import ConditionalConvBlocks


class VariationalVariancePredictor(nn.Module):

    def __init__(self,
                 c_latent,  # hidden size of what goes in
                 hidden_size,  # hidden size internal in this module
                 out_channels,  # how many dimensions the vectors in the output sequence should have (for a variance predictor, that's 1)
                 kernel_size,  # kernel size of the convolutions
                 n_layers,  # how deep to embed
                 c_cond=0,
                 strides=[4],
                 norm_type="ln",
                 spk_emb_size=256):
        super().__init__()
        self.strides = strides
        self.hidden_size = hidden_size
        self.latent_size = c_latent
        self.pre_net = nn.Sequential(*[nn.ConvTranspose1d(c_latent, hidden_size, kernel_size=s, stride=s) if i == 0 else
                                       nn.ConvTranspose1d(hidden_size, hidden_size, kernel_size=s, stride=s) for i, s in
                                       enumerate(strides)])
        self.nn = ConditionalConvBlocks(hidden_size, c_cond, hidden_size, [1] * n_layers, kernel_size,
                                        layers_in_block=2, is_BTC=False, norm_type=norm_type, spk_emb_size=spk_emb_size)
        self.out_proj = nn.Conv1d(hidden_size, out_channels, 1)

    def forward(self, cond, nonpadding=None, utt_emb=None):
        # first we sample random noise
        x = torch.randn([cond.shape[0], self.latent_size, cond.shape[2]]).to(cond.device)
        # then we pass the random noise through a pre net
        x = self.pre_net(x)
        # afterwards we have to mask out all the padding values again
        if nonpadding is None:
            mask_padding = 1
        else:
            mask_padding = nonpadding
        x = x * mask_padding
        # then we pass the shifted noise through a network that embeds it
        x = self.nn(x, nonpadding=nonpadding, cond=cond, utt_emb=utt_emb) * mask_padding
        # finally we project each vector in the sequence into a single value
        x = self.out_proj(x)
        return x
