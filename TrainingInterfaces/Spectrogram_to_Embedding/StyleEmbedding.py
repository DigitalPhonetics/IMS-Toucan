import numpy
import torch

from .swin.build import build_model


class StyleEmbedding(torch.nn.Module):
    """
    The style embedding should provide information of the speaker and their speaking style

    The feedback signal for the module will come from the TTS objective, so it doesn't have a dedicated train loop.
    Optionally we could pretrain the module on the speaker identification task.
    """

    def __init__(self, swin_config=None):
        super().__init__()
        # SWIN architecture
        if not swin_config:
            self.swin_config = {
                "model_type": "swin",
                "img_size": [256, 80],
                "patch_size": [16, 10],
                "in_chans": 1,
                "num_classes": 64,
                "embed_dim": 64,
                "depths": [2, 2, 2], #[2, 2, 18, 2],
                "num_heads": [2, 4, 8],
                "window_size": 4,
                "mlp_ratio": 4,
                "qkv_bias": False,
                "qk_scale": None,
                "drop_rate": 0,
                "drop_path_rate": 0.3,
                "ape"           : False,
                "patch_norm"    : True,
                "use_checkpoint": False
                }
        else:
            self.swin_config = swin_config

        self.swin = build_model(self.swin_config)
        n_parameters = sum(p.numel() for p in self.swin.parameters() if p.requires_grad)
        print('SWIN number of params:', n_parameters)

    def forward(self, batch_of_spectrograms, batch_of_spectrogram_lengths):
        """
        Args:
            batch_of_spectrograms: b is the batch axis, 80 features per timestep
                                   and l time-steps, which may include padding
                                   for most elements in the batch (b, l, 80)
            batch_of_spectrogram_lengths: indicate for every element in the batch,
                                          what the true length is, since they are
                                          all padded to the length of the longest
                                          element in the batch (b, 1)

        Returns:
            batch of 64 dimensional embeddings (b,64)
        """

        # we take a random window with a length of 256 out of the spectrogram or add random zero padding in front and back to get a length of 256
        window_size = 256
        list_of_specs = list()
        for index, spec_length in enumerate(batch_of_spectrogram_lengths):
            spec = batch_of_spectrograms[index][:spec_length]
            if spec_length > window_size:
                # take random window
                frames_to_remove = spec_length - window_size
                remove_front = numpy.random.randint(low=0, high=frames_to_remove.squeeze().cpu())  # [0]
                list_of_specs.append(spec[remove_front:remove_front + window_size])
            elif spec_length < window_size:
                # add random padding
                frames_to_pad = window_size - spec_length
                pad_front = numpy.random.randint(low=0, high=frames_to_pad.squeeze().cpu())  # [0]
                list_of_specs.append(torch.nn.functional.pad(input=spec, pad=(0, 0, int(pad_front), frames_to_pad.cpu() - pad_front)))
            elif spec_length == window_size:
                # take as is
                list_of_specs.append(spec)

        batch_of_spectrograms_unified_length = torch.stack(list_of_specs, dim=0)

        batch_of_spectrograms_unified_length = batch_of_spectrograms_unified_length.view(
            batch_of_spectrograms_unified_length.size(0),
            1,
            batch_of_spectrograms_unified_length.size(1),
            batch_of_spectrograms_unified_length.size(2),
        )

        speaker_embedding = self.swin(batch_of_spectrograms_unified_length)
        return speaker_embedding
