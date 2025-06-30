"""
Copied from https://github.com/KdaiP/StableTTS by https://github.com/KdaiP

https://github.com/KdaiP/StableTTS/blob/eebb177ebf195fd1246dedabec4ef69d9351a4f8/models/flow_matching.py

Code is under MIT License
"""

import imageio
import torch
import torch.nn.functional as F

from Modules.ToucanTTS.dit_wrapper import Decoder
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
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, c=c), z
        #return self.solve_heun(z, t_span=t_span, mu=mu, mask=mask, c=c), z

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

    def compute_loss(self, x1, mask, mu, c, x0=None):
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
        if x0 is None:
            z = torch.randn_like(x1)
        else:
            z = x0

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        loss = F.mse_loss(self.estimator(y, mask, mu, t.squeeze(), c),
                          u,
                          reduction="sum") / (torch.sum(mask) * u.shape[1])
        return loss, y
    
    def compute_loss_log(self, x1, mask, mu, c):
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
        x1 = torch.exp(x1) - 1
        x1 = x1.int().float()
        b, _, t = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z
        
        int_y = torch.exp(self.estimator(y, mask, mu, t.squeeze(), c)) - 1
        int_y = int_y.int().float()
        loss = F.mse_loss(int_y,
                          u,
                          reduction="sum") / (torch.sum(mask) * u.shape[1])
        return loss, y

    def solve_heun(self, x, t_span, mu, mask, c=None, prompt=None, training=False, guidance_scale=1.0,
                    input_lens=None):
        """
        Fixed heun solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            c: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # -! : reserved space for debugger
        sol = []
        steps = 1
        if x is None:
            x = torch.randn_like(mu)

        while steps <= len(t_span) - 1:
            dphi_dt = self.func_dphi_dt_fast(x, mask, mu, t, c=c, prompt=prompt, training=training,
                                             guidance_scale=guidance_scale, input_lens=input_lens)
            dphi_dt_2 = self.func_dphi_dt_fast(x + dt * dphi_dt, mask, mu, t + dt, c=c, prompt=prompt,
                                               training=training, guidance_scale=guidance_scale, input_lens=input_lens)
            x = x + dt * 0.5 * (dphi_dt + dphi_dt_2)
            t = t + dt

            sol.append(x)
            if steps < len(t_span) - 1:
                dt = t_span[steps + 1] - t
            steps += 1

        return sol[-1]
    
    def func_dphi_dt(self, x, mask, mu, t, c=None, prompt=None, training=False, guidance_scale=1.0, input_lens=None):
        if prompt is None:
                dphi_dt = self.estimator(x, mask, mu, t, c=c)
        else:
            dphi_dt = self.estimator(x, mask, mu, t, c=c, prompt=prompt, training=training, input_lens=input_lens)

        if type(guidance_scale) == float and guidance_scale > 0.0:
            mu_avg = mu.mean(2, keepdims=True).expand_as(mu)
            if prompt is None:
                dphi_avg = self.estimator(x, mask, mu_avg, t, c=c)
            else:
                dphi_avg = self.estimator(x, mask, mu_avg, t, c=c, prompt=prompt, training=training, input_lens=input_lens)
            dphi_dt = dphi_dt + guidance_scale * (dphi_dt - dphi_avg)
        elif torch.is_tensor(guidance_scale) and torch.sum(guidance_scale) > 0.0:
            mu_avg = mu.mean(2, keepdims=True).expand_as(mu)
            if prompt is None:
                dphi_avg = self.estimator(x, mask, mu_avg, t, c=c)
            else:
                dphi_avg = self.estimator(x, mask, mu_avg, t, c=c, prompt=prompt, training=training, input_lens=input_lens)
            dphi_dt = dphi_dt + guidance_scale.unsqueeze(-1).unsqueeze(-1) * (dphi_dt - dphi_avg)

        return dphi_dt

    def func_dphi_dt_fast(self, x, mask, mu, t, c=None, prompt=None, training=False, guidance_scale=1.0,
                          input_lens=None, cfg_mask_mode=False):
        if (type(guidance_scale) == float and guidance_scale > 0.0) or (
                torch.is_tensor(guidance_scale) and torch.sum(guidance_scale) > 0.0):
            b = x.shape[0]

            # one batch for both c/unc
            new_mu = torch.cat([mu, mu.mean(2, keepdims=True).expand_as(mu)], dim=0)
            if cfg_mask_mode:
                new_mu[b:] = 0.
            if torch.is_tensor(c):
                new_c = c.repeat(2, 1)
                if cfg_mask_mode:
                    new_c[b:] = 0.
            else:
                new_c = c
            if torch.is_tensor(prompt):
                new_prompt = prompt.repeat(2, 1, 1)
                if cfg_mask_mode:
                    new_prompt[b:] = 0.
            else:
                new_prompt = prompt
            if prompt is None:
                dphi_dt_joint = self.estimator(
                    x.repeat(2, 1, 1),
                    mask.repeat(2, 1, 1),
                    new_mu,
                    t if t.dim() == 0 else t.repeat(2),
                    c=new_c)
            else:
                dphi_dt_joint = self.estimator(
                    x.repeat(2, 1, 1),
                    mask.repeat(2, 1, 1),
                    new_mu,
                    t if t.dim() == 0 else t.repeat(2),
                    c=new_c,
                    prompt=new_prompt,
                    training=training,
                    input_lens=input_lens.repeat(2, 1) if torch.is_tensor(input_lens) else (
                        input_lens * 2 if type(input_lens) == list else input_lens)
                )

            if (type(guidance_scale) == float and guidance_scale > 0.0):
                dphi_dt = (1.0 + guidance_scale) * dphi_dt_joint[:b] - guidance_scale * dphi_dt_joint[b:]
            else:
                dphi_dt = (1.0 + guidance_scale.unsqueeze(-1).unsqueeze(-1)) * dphi_dt_joint[:b] - guidance_scale.unsqueeze(-1).unsqueeze(-1) * dphi_dt_joint[b:]

            return dphi_dt

        return self.func_dphi_dt(x, mask, mu, t, c=c, prompt=prompt, training=training,
                                 guidance_scale=guidance_scale, input_lens=input_lens)

def create_plot_of_all_solutions(sol):
    gif_collector = list()
    for step_index, solution in enumerate(sol):
        unbatched_solution = solution[0]  # remove the batch axis (if there are more than one element in the batch, we only take the first)
        plot_spec_tensor(unbatched_solution, "tmp", step_index, title=step_index + 1)
        gif_collector.append(imageio.v2.imread(f"tmp/{step_index}.png"))
    for _ in range(10):
        gif_collector.append(gif_collector[-1])
    imageio.mimsave("tmp/animation.gif", gif_collector, fps=6, loop=0)
