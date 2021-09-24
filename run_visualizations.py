"""
A place for demos, visualizations and sanity checks
"""

import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torchviz

from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2
from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.Tacotron2 import Tacotron2


def plot_fastspeech_architecture():
    text = torch.LongTensor([1, 2, 3, 4])
    speech = torch.zeros(80, 50)
    durations = torch.LongTensor([1, 2, 3, 4])
    pitch = torch.Tensor([1.0]).unsqueeze(0)
    energy = torch.Tensor([1.0]).unsqueeze(0)
    model = FastSpeech2(idim=166, odim=80, spk_embed_dim=None)
    out = model.inference(text=text, speech=speech, durations=durations, pitch=pitch,
                          energy=energy, speaker_embeddings=None, use_teacher_forcing=True)
    torchviz.make_dot(out, dict(model.named_parameters())).render("fastspeech2_graph", format="png")


def plot_tacotron2_architecture():
    model = Tacotron2(idim=166, odim=80, spk_embed_dim=10)
    out = model.inference(text=torch.LongTensor([1, 2, 3, 4]), speech=None, speaker_embeddings=torch.zeros(10), use_teacher_forcing=False)
    torchviz.make_dot(out, dict(model.named_parameters())).render("tacotron2_graph", format="png")


def show_audio_lens_in_dataset(path_list):
    lens = list()
    for path in path_list:
        wave, sr = sf.read(path)
        lens.append(round(len(wave) / sr))
    lens.sort()
    plt.hist(lens)
    plt.show()


def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def show_all_models_params():
    from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.Tacotron2 import Tacotron2
    from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2
    model = Tacotron2(idim=166, odim=80)
    print("Number of (trainable) Parameters in Tacotron2: {}".format(count_parameters(model)))
    model = FastSpeech2(idim=166, odim=80)
    print("Number of (trainable) Parameters in FastSpeech2: {}".format(count_parameters(model)))


if __name__ == '__main__':
    plot_tacotron2_architecture()
    show_all_models_params()
