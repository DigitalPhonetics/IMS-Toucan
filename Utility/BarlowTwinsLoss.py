# adapted from https://github.com/facebookresearch/barlowtwins

import torch


class BarlowTwinsLoss(torch.nn.Module):

    def __init__(self, lambd=1e-5, vector_dimensions=64):
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
