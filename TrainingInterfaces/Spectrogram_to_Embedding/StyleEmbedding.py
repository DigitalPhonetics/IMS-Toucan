import torch


class StyleEmbedding(torch.nn.Module):
    """
    The style embedding should provide information of the speaker and their speaking style

    The feedback signal for the module will come from the TTS objective, so it doesn't have a dedicated train loop.
    Optionally we could pretrain the module on the speaker identification task.
    """

    def __init__(self):
        super().__init__()
        # SWIN architecture

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
        return torch.zeros((batch_of_spectrograms.size(0), 64))
