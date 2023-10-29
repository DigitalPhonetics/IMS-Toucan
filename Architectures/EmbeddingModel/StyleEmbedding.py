import torch

from Architectures.EmbeddingModel.GST import GSTStyleEncoder
from Architectures.EmbeddingModel.StyleTTSEncoder import StyleEncoder as StyleTTSEncoder


class StyleEmbedding(torch.nn.Module):
    """
    The style embedding should provide information of the speaker and their speaking style

    The feedback signal for the module will come from the TTS objective, so it doesn't have a dedicated train loop.
    The train loop does however supply supervision in the form of a barlow twins objective.

    See the git history for some other approaches for style embedding, like the SWIN transformer
    and a simple LSTM baseline. GST turned out to be the best.
    """

    def __init__(self, embedding_dim=16, style_tts_encoder=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_gst = not style_tts_encoder
        if style_tts_encoder:
            self.style_encoder = StyleTTSEncoder(style_dim=embedding_dim)
        else:
            self.style_encoder = GSTStyleEncoder(gst_token_dim=embedding_dim)

    def forward(self,
                batch_of_feature_sequences,
                batch_of_feature_sequence_lengths):
        """
        Args:
            batch_of_feature_sequences: b is the batch axis, 128 features per timestep
                                   and l time-steps, which may include padding
                                   for most elements in the batch (b, l, 128)
            batch_of_feature_sequence_lengths: indicate for every element in the batch,
                                          what the true length is, since they are
                                          all padded to the length of the longest
                                          element in the batch (b, 1)
        Returns:
            batch of n dimensional embeddings (b,n)
        """

        minimum_sequence_length = 512
        specs = list()
        for index, spec_length in enumerate(batch_of_feature_sequence_lengths):
            spec = batch_of_feature_sequences[index][:spec_length]
            # double the length at least once, then check
            spec = spec.repeat((2, 1))
            current_spec_length = len(spec)
            while current_spec_length < minimum_sequence_length:
                # make it longer
                spec = spec.repeat((2, 1))
                current_spec_length = len(spec)
            specs.append(spec[:minimum_sequence_length])

        spec_batch = torch.stack(specs, dim=0)
        return self.style_encoder(speech=spec_batch)


if __name__ == '__main__':
    style_emb = StyleEmbedding(style_tts_encoder=False)
    print(f"GST parameter count: {sum(p.numel() for p in style_emb.style_encoder.parameters() if p.requires_grad)}")

    seq_length = 398
    print(style_emb(torch.randn(5, seq_length, 512),
                    torch.tensor([seq_length, seq_length, seq_length, seq_length, seq_length])).shape)

    style_emb = StyleEmbedding(style_tts_encoder=True)
    print(f"StyleTTS encoder parameter count: {sum(p.numel() for p in style_emb.style_encoder.parameters() if p.requires_grad)}")

    seq_length = 398
    print(style_emb(torch.randn(5, seq_length, 512),
                    torch.tensor([seq_length, seq_length, seq_length, seq_length, seq_length])).shape)
