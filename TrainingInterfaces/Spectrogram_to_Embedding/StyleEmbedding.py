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

    def forward(self,
                batch_of_spectrograms,
                batch_of_spectrogram_lengths,
                return_all_outs=False,
                return_only_refs=False):
        """
        Args:
            return_only_refs: return reference embedding instead of mixed style tokens
            batch_of_spectrograms: b is the batch axis, 80 features per timestep
                                   and l time-steps, which may include padding
                                   for most elements in the batch (b, l, 80)
            batch_of_spectrogram_lengths: indicate for every element in the batch,
                                          what the true length is, since they are
                                          all padded to the length of the longest
                                          element in the batch (b, 1)
            return_all_outs: boolean indicating whether the output will be used for a feature matching loss
        Returns:
            batch of 256 dimensional embeddings (b,256)
        """

        minimum_sequence_length = 812
        specs = list()
        for index, spec_length in enumerate(batch_of_spectrogram_lengths):
            spec = batch_of_spectrograms[index][:spec_length]
            # double the length at least once, then check
            spec = spec.repeat((2, 1))
            current_spec_length = len(spec)
            while current_spec_length < minimum_sequence_length:
                # make it longer
                spec = spec.repeat((2, 1))
                current_spec_length = len(spec)
            specs.append(spec[:812])

        spec_batch = torch.stack(specs, dim=0)
        return self.gst(speech=spec_batch,
                        return_all_outs=return_all_outs,
                        return_only_ref=return_only_refs)


if __name__ == '__main__':
    style_emb = StyleEmbedding()
    print(f"GST parameter count: {sum(p.numel() for p in style_emb.gst.parameters() if p.requires_grad)}")

    seq_length = 398
    print(style_emb(torch.randn(5, seq_length, 80),
                    torch.tensor([seq_length, seq_length, seq_length, seq_length, seq_length]),
                    return_only_refs=False).shape)
