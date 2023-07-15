"""
MIT License

Copyright (c) 2023 Hugo Flores García and Prem Seetharaman

https://github.com/hugofloresgarcia/vampnet

Modified by Florian Lux 2023
"""

import torch.nn as nn
from torch.nn.utils import weight_norm


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))
