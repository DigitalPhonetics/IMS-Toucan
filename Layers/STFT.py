import torch
from torch_complex.tensor import ComplexTensor

from Utility.utils import make_pad_mask


class STFT(torch.nn.Module):
    def __init__(self,
                 n_fft: int = 512,
                 win_length: int = None,
                 hop_length: int = 128,
                 window="hann",
                 center: bool = True,
                 normalized: bool = False,
                 onesided: bool = True):
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

    def forward(self, input: torch.Tensor, ilens: torch.Tensor = None):
        """
        STFT forward function.
        Args:
            input: (Batch, Nsamples) or (Batch, Nsample, Channels)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, Freq, 2) or (Batch, Frames, Channels, Freq, 2)
        """
        bs = input.size(0)
        if input.dim() == 3:
            multi_channel = True
            # input: (Batch, Nsample, Channels) -> (Batch * Channels, Nsample)
            input = input.transpose(1, 2).reshape(-1, input.size(1))
        else:
            multi_channel = False

        # output: (Batch, Freq, Frames, 2=real_imag)
        # or (Batch, Channel, Freq, Frames, 2=real_imag)
        if self.window is not None:
            window_func = getattr(torch, f"{self.window}_window")
            window = window_func(self.win_length, dtype=input.dtype, device=input.device)
        else:
            window = None
        output = torch.stft(input,
                            n_fft=self.n_fft,
                            win_length=self.win_length,
                            hop_length=self.hop_length,
                            center=self.center,
                            window=window,
                            normalized=self.normalized,
                            onesided=self.onesided)
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

            olens = (ilens - self.win_length) // self.hop_length + 1
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

        wavs = istft(input,
                     n_fft=self.n_fft,
                     hop_length=self.hop_length,
                     win_length=self.win_length,
                     window=window,
                     center=self.center,
                     normalized=self.normalized,
                     onesided=self.onesided,
                     length=ilens.max() if ilens is not None else ilens)

        return wavs, ilens
