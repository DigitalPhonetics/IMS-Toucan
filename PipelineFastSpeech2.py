import torch
import torchviz

from FastSpeech.FastSpeech2 import FastSpeech2


def featurize_corpus(path_to_corpus):
    # load pair of text and speech
    # apply collect_features()
    # store features in Dataset dict
    # repeat for all pairs
    # Dump dict to file
    pass


def collect_features(text, wave):
    # return: pitch, energy, speech features, text features, durations, speaker embeddings
    pass


def train_loop():
    pass


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def show_model(model):
    print(model)
    print("\n\nNumber of Parameters: {}".format(count_parameters(model)))


def plot_model():
    fast = FastSpeech2(idim=131, odim=256)
    out = fast(text_tensors=torch.randint(high=120, size=(1, 23)),
               text_lengths=torch.tensor([23]),
               gold_speech=torch.rand((1, 1234, 256)),
               speech_lengths=torch.tensor([1234]),
               gold_durations=torch.tensor([[1]]),
               durations_lengths=torch.tensor([1]),
               gold_pitch=torch.tensor([[1]]),
               pitch_lengths=torch.tensor([1]),
               gold_energy=torch.tensor([[1]]),
               energy_lengths=torch.tensor([1]),
               spembs=torch.rand(256).unsqueeze(0))
    torchviz.make_dot(out[0].mean(), dict(fast.named_parameters())).render("fastspeech2_graph", format="png")


if __name__ == '__main__':
    pass
