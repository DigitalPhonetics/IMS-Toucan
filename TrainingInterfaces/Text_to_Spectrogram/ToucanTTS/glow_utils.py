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


def squeeze(x, nonpadding=None, n_sqz=2):
    b, c, t = x.size()

    t = (t // n_sqz) * n_sqz
    x = x[:, :, :t]
    x_sqz = x.view(b, c, t // n_sqz, n_sqz)
    x_sqz = x_sqz.permute(0, 3, 1, 2).contiguous().view(b, c * n_sqz, t // n_sqz)

    if nonpadding is not None:
        nonpadding = nonpadding[:, :, n_sqz - 1::n_sqz]
    else:
        nonpadding = torch.ones(b, 1, t // n_sqz).to(device=x.device, dtype=x.dtype)
    return x_sqz * nonpadding, nonpadding


def unsqueeze(x, nonpadding=None, n_sqz=2):
    b, c, t = x.size()

    x_unsqz = x.view(b, n_sqz, c // n_sqz, t)
    x_unsqz = x_unsqz.permute(0, 2, 3, 1).contiguous().view(b, c // n_sqz, t * n_sqz)

    if nonpadding is not None:
        nonpadding = nonpadding.unsqueeze(-1).repeat(1, 1, 1, n_sqz).view(b, 1, t * n_sqz)
    else:
        nonpadding = torch.ones(b, 1, t * n_sqz).to(device=x.device, dtype=x.dtype)
    return x_unsqz * nonpadding, nonpadding
