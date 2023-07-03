"""
MIT License

Copyright (c) 2022 Yi Ren

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
from torch import nn


def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class WN(torch.nn.Module):

    def __init__(self, hidden_size, kernel_size, dilation_rate, n_layers, c_cond=0,
                 p_dropout=0, share_cond_layers=False, is_BTC=False, use_weightnorm=False):
        """
        If weightnorm is set to false, we can deepcopy the module, which we need to be able to do to perform SWA.
        Without weightnorm, the module will probably take a little longer to converge.
        """
        super(WN, self).__init__()
        assert (kernel_size % 2 == 1)
        assert (hidden_size % 2 == 0)
        self.is_BTC = is_BTC
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = c_cond
        self.p_dropout = p_dropout
        self.share_cond_layers = share_cond_layers

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if c_cond != 0 and not share_cond_layers:
            cond_layer = torch.nn.Conv1d(c_cond, 2 * hidden_size * n_layers, 1)
            if use_weightnorm:
                self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')
            else:
                self.cond_layer = cond_layer

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(hidden_size, 2 * hidden_size, kernel_size,
                                       dilation=dilation, padding=padding)
            if use_weightnorm:
                in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_size
            else:
                res_skip_channels = hidden_size

            res_skip_layer = torch.nn.Conv1d(hidden_size, res_skip_channels, 1)
            if use_weightnorm:
                res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, nonpadding=None, cond=None):
        if self.is_BTC:
            x = x.transpose(1, 2)
            cond = cond.transpose(1, 2) if cond is not None else None
            nonpadding = nonpadding.transpose(1, 2) if nonpadding is not None else None
        if nonpadding is None:
            nonpadding = 1
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_size])

        if cond is not None and not self.share_cond_layers:
            cond = self.cond_layer(cond)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            x_in = self.drop(x_in)
            if cond is not None:
                cond_offset = i * 2 * self.hidden_size
                cond_l = cond[:, cond_offset:cond_offset + 2 * self.hidden_size, :]
            else:
                cond_l = torch.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(x_in, cond_l, n_channels_tensor)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = (x + res_skip_acts[:, :self.hidden_size, :]) * nonpadding
                output = output + res_skip_acts[:, self.hidden_size:, :]
            else:
                output = output + res_skip_acts
        output = output * nonpadding
        if self.is_BTC:
            output = output.transpose(1, 2)
        return output

    def remove_weight_norm(self):
        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(remove_weight_norm)
