# adapted from https://github.com/facebookresearch/barlowtwins

from math import exp

import torch
import torch.nn.functional as F
from torch.autograd import Variable


class BarlowTwinsLoss(torch.nn.Module):

    def __init__(self, lambd=1e-5, vector_dimensions=256):
        super().__init__()
        self.lambd = lambd
        self.bn = torch.nn.BatchNorm1d(vector_dimensions, affine=False)

    def forward(self, z1, z2):
        c = self.bn(z1).T @ self.bn(z2)
        c.div_(z1.size(0))
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class TripletLoss(torch.nn.Module):

    def __init__(self, margin):
        super().__init__()
        self.cosine_similarity = torch.nn.CosineSimilarity()
        self.margin = margin

    def forward(self,
                anchor_embeddings,
                positive_embeddings,
                negative_embeddings):
        positive_distance = 1 - self.cosine_similarity(anchor_embeddings, positive_embeddings)
        negative_distance = 1 - self.cosine_similarity(anchor_embeddings, negative_embeddings)

        losses = torch.max(positive_distance - negative_distance + self.margin,
                           torch.full_like(positive_distance, 0))
        return torch.mean(losses)


# The following is taken from https://github.com/NATSpeech/NATSpeech/blob/aef3aa8899c82e40a28e4f59d559b46b18ba87e8/utils/metrics/ssim.py

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1)


class SSIM(torch.nn.Module):
    """
    Adapted from https://github.com/Po-Hsun-Su/pytorch-ssim
    """

    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


window = None


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    global window
    if window is None:
        window = create_window(window_size, channel)
        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)
