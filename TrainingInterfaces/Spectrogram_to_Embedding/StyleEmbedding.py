import numpy
import torch

from TrainingInterfaces.Spectrogram_to_Embedding.GST import StyleEncoder


class StyleEmbedding(torch.nn.Module):
    """
    The style embedding should provide information of the speaker and their speaking style

    The feedback signal for the module will come from the TTS objective, so it doesn't have a dedicated train loop.
    The train loop does however supply supervision in the form of a barlow twins objective.
    """

    def __init__(self):
        super().__init__()
        self.gst = StyleEncoder()

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
                remove_front = numpy.random.randint(low=0, high=frames_to_remove.cpu().item())
                list_of_specs.append(spec[remove_front:remove_front + window_size])
            elif spec_length < window_size:
                # add random padding
                frames_to_pad = window_size - spec_length
                pad_front = numpy.random.randint(low=0, high=frames_to_pad.cpu().item())
                list_of_specs.append(torch.nn.functional.pad(input=spec, pad=(0, 0, int(pad_front), frames_to_pad.cpu() - pad_front)))
            elif spec_length == window_size:
                # take as is
                list_of_specs.append(spec)

        batch_of_spectrograms_unified_length = torch.stack(list_of_specs, dim=0)
        return self.gst(batch_of_spectrograms_unified_length)


if __name__ == '__main__':
    style_emb = StyleEmbedding()
    print(f"GST parameter count: {sum(p.numel() for p in style_emb.gst.parameters() if p.requires_grad)}")
    print(style_emb(torch.randn(5, 600, 80), torch.tensor([600, 600, 600, 600, 600])).shape)
