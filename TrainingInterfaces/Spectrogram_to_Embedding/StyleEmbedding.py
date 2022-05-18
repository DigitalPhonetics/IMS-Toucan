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
                "img_size": 224,
                "patch_size": 4,
                "in_chans": 1,
                "num_classes": 64,
                "embed_dim": 128,
                "depths": [2, 2, 18, 2],
                "num_heads": [4, 8, 16, 32],
                "window_size": 7,
                "mlp_ratio": 4,
                "qkv_bias": False,
                "qk_scale": None,
                "drop_rate": 0,
                "drop_path_rate": 0.3,
                "ape": False,
                "patch_norm": True,
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
                                   for most elements in the batch (b, 80, l)
            batch_of_spectrogram_lengths: indicate for every element in the batch,
                                          what the true length is, since they are
                                          all padded to the length of the longest
                                          element in the batch (b, 1)

        Returns:
            batch of 64 dimensional embeddings (b,64)
        """
        batch_of_spectrograms = batch_of_spectrograms.view(
            batch_of_spectrograms.size(0),
            1,
            batch_of_spectrograms.size(1),
            batch_of_spectrograms.size(2),
        )
        
        speaker_embedding = self.swin(batch_of_spectrograms)
        return speaker_embedding
