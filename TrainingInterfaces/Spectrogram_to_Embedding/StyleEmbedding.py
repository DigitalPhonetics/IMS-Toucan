import torch

from TrainingInterfaces.Spectrogram_to_Embedding.GST import StyleEncoder


class StyleEmbedding(torch.nn.Module):
    """
    The style embedding should provide information of the speaker and their speaking style

    The feedback signal for the module will come from the TTS objective, so it doesn't have a dedicated train loop.
    The train loop does however supply supervision in the form of a barlow twins objective.

    See the git history for some other approaches for style embedding, like the SWIN transformer
    and a simple LSTM baseline. GST turned out to be the best.
    """

    def __init__(self):
        super().__init__()
        self.gst = StyleEncoder()

    def forward(self, batch_of_spectrograms):
        """
        Args:
            batch_of_spectrograms: b is the batch axis, 80 features per timestep
                                   and l time-steps, which may include padding
                                   for most elements in the batch (b, l, 80)
        Returns:
            batch of 256 dimensional embeddings (b,256)
        """
        return self.gst(batch_of_spectrograms)


if __name__ == '__main__':
    style_emb = StyleEmbedding()
    print(f"GST parameter count: {sum(p.numel() for p in style_emb.gst.parameters() if p.requires_grad)}")
    print(style_emb(torch.randn(5, 600, 80)).shape)
