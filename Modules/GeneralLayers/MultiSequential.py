# Written by Shigeki Karita, 2019
# Published under Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Adapted by Florian Lux, 2021

import torch


class MultiSequential(torch.nn.Sequential):
    """
    Multi-input multi-output torch.nn.Sequential.
    """

    def forward(self, *args):
        """
        Repeat.
        """
        for m in self:
            args = m(*args)
        return args


def repeat(N, fn):
    """
    Repeat module N times.

    Args:
        N (int): Number of repeat time.
        fn (Callable): Function to generate module.

    Returns:
        MultiSequential: Repeated model instance.
    """
    return MultiSequential(*[fn(n) for n in range(N)])
