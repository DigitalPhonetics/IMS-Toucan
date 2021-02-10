#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch


class Swish(torch.nn.Module):
    """
    Construct an Swish activation function for Conformer.
    """

    def forward(self, x):
        """
        Return Swish activation function.
        """
        return x * torch.sigmoid(x)
