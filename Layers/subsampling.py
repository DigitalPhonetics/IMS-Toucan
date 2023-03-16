import torch
import torch.nn as nn


class Conv2dSubsampling(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Convert an input of shape [N, T, idim] to an output
    with shape [N, T', odim], where
    T' = ((T-1)//2 - 1)//2, which approximates T' == T//4

    It is based on
    https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/subsampling.py  # noqa
    """

    def __init__(self, idim: int, odim: int) -> None:
        """
        Args:
          idim:
            Input dim. The input shape is [N, T, idim].
            Caution: It requires: T >=7, idim >=7
          odim:
            Output dim. The output shape is [N, ((T-1)//2 - 1)//2, odim]
        """
        assert idim >= 7
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=odim, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=odim, out_channels=odim, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        self.out = nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Subsample x.

        Args:
          x:
            Its shape is [N, T, idim].

        Returns:
          Return a tensor of shape [N, ((T-1)//2 - 1)//2, odim]
        """
        # On entry, x is [N, T, idim]
        x = x.unsqueeze(1)  # [N, T, idim] -> [N, 1, T, idim] i.e., [N, C, H, W]
        x = self.conv(x)
        # Now x is of shape [N, odim, ((T-1)//2 - 1)//2, ((idim-1)//2 - 1)//2]
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        # Now x is of shape [N, ((T-1)//2 - 1))//2, odim]
        return x


class VggSubsampling(nn.Module):
    """Trying to follow the setup described in the following paper:
    https://arxiv.org/pdf/1910.09799.pdf

    This paper is not 100% explicit so I am guessing to some extent,
    and trying to compare with other VGG implementations.

    Convert an input of shape [N, T, idim] to an output
    with shape [N, T', odim], where
    T' = ((T-1)//2 - 1)//2, which approximates T' = T//4
    """

    def __init__(self, idim: int, odim: int) -> None:
        """Construct a VggSubsampling object.

        This uses 2 VGG blocks with 2 Conv2d layers each,
        subsampling its input by a factor of 4 in the time dimensions.

        Args:
          idim:
            Input dim. The input shape is [N, T, idim].
            Caution: It requires: T >=7, idim >=7
          odim:
            Output dim. The output shape is [N, ((T-1)//2 - 1)//2, odim]
        """
        super().__init__()

        cur_channels = 1
        layers = []
        block_dims = [32, 64]

        # The decision to use padding=1 for the 1st convolution, then padding=0
        # for the 2nd and for the max-pooling, and ceil_mode=True, was driven by
        # a back-compatibility concern so that the number of frames at the
        # output would be equal to:
        #  (((T-1)//2)-1)//2.
        # We can consider changing this by using padding=1 on the
        # 2nd convolution, so the num-frames at the output would be T//4.
        for block_dim in block_dims:
            layers.append(
                torch.nn.Conv2d(
                    in_channels=cur_channels,
                    out_channels=block_dim,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                )
            )
            layers.append(torch.nn.ReLU())
            layers.append(
                torch.nn.Conv2d(
                    in_channels=block_dim,
                    out_channels=block_dim,
                    kernel_size=3,
                    padding=0,
                    stride=1,
                )
            )
            layers.append(
                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
            )
            cur_channels = block_dim

        self.layers = nn.Sequential(*layers)

        self.out = nn.Linear(block_dims[-1] * (((idim - 1) // 2 - 1) // 2), odim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Subsample x.

        Args:
          x:
            Its shape is [N, T, idim].

        Returns:
          Return a tensor of shape [N, ((T-1)//2 - 1)//2, odim]
        """
        x = x.unsqueeze(1)
        x = self.layers(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        return x