import torch.nn as nn


class DepthwiseSeparableConvolution(nn.Module):

    def __init__(self, n_in, n_out, kernel_size=3, padding=1, bias=False):
        super(DepthwiseSeparableConvolution, self).__init__()
        self.depthwise = nn.Conv2d(n_in, n_in, kernel_size=kernel_size, padding=padding, groups=n_in, bias=bias)
        self.pointwise = nn.Conv2d(n_in, n_out, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
