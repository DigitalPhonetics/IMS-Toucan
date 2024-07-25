"""
Taken from ESPNet
"""

import torch
from torch.functional import stft as torch_stft
from torch_complex.tensor import ComplexTensor

from Utility.utils import make_pad_mask


class STFT(torch.nn.Module):

    def __init__(self, n_fft=512,
                 win_length=None,
                 hop_length=128,
                 window="hann",
                 center=True,
                 normalized=False,
                 onesided=True):
        super().__init__()
        self.n_fft = n_fft
        if win_length is None:
            self.win_length = n_fft
        else:
            self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        self.window = window

    def extra_repr(self):
        return (f"n_fft={self.n_fft}, "
                f"win_length={self.win_length}, "
                f"hop_length={self.hop_length}, "
                f"center={self.center}, "
                f"normalized={self.normalized}, "
                f"onesided={self.onesided}")

    def forward(self, input_wave, ilens=None):
        """
        STFT forward function.
        Args:
            input_wave: (Batch, Nsamples) or (Batch, Nsample, Channels)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, Freq, 2) or (Batch, Frames, Channels, Freq, 2)
        """
        bs = input_wave.size(0)

        if input_wave.dim() == 3:
            multi_channel = True
            # input: (Batch, Nsample, Channels) -> (Batch * Channels, Nsample)
            input_wave = input_wave.transpose(1, 2).reshape(-1, input_wave.size(1))
        else:
            multi_channel = False

        # output: (Batch, Freq, Frames, 2=real_imag)
        # or (Batch, Channel, Freq, Frames, 2=real_imag)
        if self.window is not None:
            window_func = getattr(torch, f"{self.window}_window")
            window = window_func(self.win_length, dtype=input_wave.dtype, device=input_wave.device)
        else:
            window = None

        complex_output = torch_stft(input=input_wave,
                                    n_fft=self.n_fft,
                                    win_length=self.win_length,
                                    hop_length=self.hop_length,
                                    center=self.center,
                                    window=window,
                                    normalized=self.normalized,
                                    onesided=self.onesided,
                                    return_complex=True)
        output = torch.view_as_real(complex_output)
        # output: (Batch, Freq, Frames, 2=real_imag)
        # -> (Batch, Frames, Freq, 2=real_imag)
        output = output.transpose(1, 2)
        if multi_channel:
            # output: (Batch * Channel, Frames, Freq, 2=real_imag)
            # -> (Batch, Frame, Channel, Freq, 2=real_imag)
            output = output.view(bs, -1, output.size(1), output.size(2), 2).transpose(1, 2)

        if ilens is not None:
            if self.center:
                pad = self.win_length // 2
                ilens = ilens + 2 * pad

            olens = torch.div((ilens - self.win_length), self.hop_length, rounding_mode='trunc') + 1
            output.masked_fill_(make_pad_mask(olens, output, 1), 0.0)
        else:
            olens = None

        return output, olens

    def inverse(self, input, ilens=None):
        """
        Inverse STFT.
        Args:
            input: Tensor(batch, T, F, 2) or ComplexTensor(batch, T, F)
            ilens: (batch,)
        Returns:
            wavs: (batch, samples)
            ilens: (batch,)
        """
        istft = torch.functional.istft

        if self.window is not None:
            window_func = getattr(torch, f"{self.window}_window")
            window = window_func(self.win_length, dtype=input.dtype, device=input.device)
        else:
            window = None

        if isinstance(input, ComplexTensor):
            input = torch.stack([input.real, input.imag], dim=-1)
        assert input.shape[-1] == 2
        input = input.transpose(1, 2)

        wavs = istft(input, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=window, center=self.center,
                     normalized=self.normalized, onesided=self.onesided, length=ilens.max() if ilens is not None else ilens)

        return wavs, ilens
