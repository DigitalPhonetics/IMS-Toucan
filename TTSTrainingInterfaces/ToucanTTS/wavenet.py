import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class WN(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=2, kernel_size=2, num_layers=8):
        super(WN, self).__init__()

        # Stack dilated convolutions with weight normalization
        self.layers = nn.ModuleList([
            weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=(kernel_size - 1) * dilation))
            for _ in range(num_layers)
        ])

    def forward(self, x):
        # Autoregressive refinement
        batch_size, channels, sequence_length = x.size()
        output = torch.zeros_like(x)

        for t in range(sequence_length):
            prediction = x[:, :, t:t + 1]

            for layer in self.layers:
                prediction = layer(prediction)

            output[:, :, t:t + 1] = prediction[:, :, -1:]  # Take the last output time step

        return output
