"""
A place for demos, visualizations and sanity checks
"""

import matplotlib.pyplot as plt
import soundfile as sf


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
    model = Tacotron2()
    print("Number of (trainable) Parameters in Tacotron2: {}".format(count_parameters(model)))
    model = FastSpeech2()
    print("Number of (trainable) Parameters in FastSpeech2: {}".format(count_parameters(model)))


if __name__ == '__main__':
    show_all_models_params()
