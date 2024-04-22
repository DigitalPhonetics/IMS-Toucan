"""
Copied from https://github.com/KdaiP/StableTTS by https://github.com/KdaiP

https://github.com/KdaiP/StableTTS/blob/eebb177ebf195fd1246dedabec4ef69d9351a4f8/models/flow_matching.py

Code is under MIT License
"""

import imageio
import torch
import torch.nn.functional as F

from Architectures.ToucanTTS.dit_wrapper import Decoder
from Utility.utils import plot_spec_tensor


# copied from https://github.com/jaywalnut310/vits/blob/main/commons.py#L121
def sequence_mask(length: torch.Tensor, max_length: int = None) -> torch.Tensor:
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


# modified from https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/components/flow_matching.py
class CFMDecoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, gin_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.gin_channels = gin_channels
        self.sigma_min = 1e-4

        self.estimator = Decoder(hidden_channels, out_channels, filter_channels, p_dropout, n_layers, n_heads, kernel_size, gin_channels)

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, c=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            c (torch.Tensor, optional): shape: (batch_size, gin_channels)

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        size = list(mu.size())
        size[1] = self.out_channels
        z = torch.randn(size=size).to(mu.device) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, c=c)

    def solve_euler(self, x, t_span, mu, mask, c, plot_solutions=False):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            c (torch.Tensor, optional): speaker condition.
                shape: (batch_size, gin_channels)
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        sol = []

        for step in range(1, len(t_span)):

            dphi_dt = self.estimator(x, mask, mu, t, c)

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        if plot_solutions:
            create_plot_of_all_solutions(sol)

        return sol[-1]

    def compute_loss(self, x1, mask, mu, c):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            c (torch.Tensor, optional): speaker condition.

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        loss = F.mse_loss(self.estimator(y, mask, mu, t.squeeze(), c),
                          u,
                          reduction="sum") / (torch.sum(mask) * u.shape[1])
        return loss, y


def create_plot_of_all_solutions(sol):
    gif_collector = list()
    for step_index, solution in enumerate(sol):
        unbatched_solution = solution[0]  # remove the batch axis (if there are more than one element in the batch, we only take the first)
        plot_spec_tensor(unbatched_solution, "tmp", step_index, title=step_index + 1)
        gif_collector.append(imageio.v2.imread(f"tmp/{step_index}.png"))
    for _ in range(10):
        gif_collector.append(gif_collector[-1])
    imageio.mimsave("tmp/animation.gif", gif_collector, fps=6, loop=0)
