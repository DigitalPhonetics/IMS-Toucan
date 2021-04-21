"""
A place for demos, visualizations and sanity checks
"""

import os

import matplotlib.pyplot as plt
import sounddevice
import soundfile as sf
import torch
import torchviz

from FastSpeech2.FastSpeech2 import FastSpeech2
from PreprocessingForTTS.ProcessAudio import AudioPreprocessor
from TransformerTTS.TransformerTTS import Transformer
from TransformerTTS.TransformerTTS import show_attention_plot


def show_att(sentence, lang="de", best_only=False):
    show_attention_plot(sentence, lang=lang, best_only=best_only)


def plot_fastspeech_architecture():
    text = torch.LongTensor([1, 2, 3, 4])
    speech = torch.zeros(80, 50)
    durations = torch.LongTensor([1, 2, 3, 4])
    pitch = torch.Tensor([1.0]).unsqueeze(0)
    energy = torch.Tensor([1.0]).unsqueeze(0)
    model = FastSpeech2(idim=133, odim=80, spk_embed_dim=None)
    out = model.inference(text=text, speech=speech, durations=durations, pitch=pitch, energy=energy, speaker_embeddings=None, use_teacher_forcing=True)
    torchviz.make_dot(out, dict(model.named_parameters())).render("fastspeech2_graph", format="pdf")


def plot_transformertts_architecture():
    text = torch.LongTensor([1, 2, 3, 4])
    speech = torch.zeros(80, 50)
    model = Transformer(idim=133, odim=80, spk_embed_dim=None)
    out = model.inference(text=text, speech=speech, speaker_embeddings=None, use_teacher_forcing=False)
    torchviz.make_dot(out, dict(model.named_parameters())).render("transformertts_graph", format="png")


def plot_melgan_training(path_to_train_loss_json="Models/Use/train_loss.json", path_to_valid_loss_json="Models/Use/valid_loss.json"):
    import matplotlib.pyplot as plt
    import json
    with open(path_to_train_loss_json, 'r') as plotting_data_file:
        train_loss_dict = json.load(plotting_data_file)
    with open(path_to_valid_loss_json, 'r') as plotting_data_file:
        valid_loss_dict = json.load(plotting_data_file)
    plt.plot(list(range(1, len(train_loss_dict["multi_res_spectral_convergence"]) + 1)), train_loss_dict["multi_res_spectral_convergence"], 'b',
             label="Spectral Distribution Loss", alpha=0.5)
    plt.plot(list(range(1, len(train_loss_dict["multi_res_log_stft_mag"]) + 1)), train_loss_dict["multi_res_log_stft_mag"], 'g',
             label="Spectral Magnitude Loss", alpha=0.5)
    plt.plot(list(range(1, len(train_loss_dict["adversarial"]) + 1)), [a / 12 for a in train_loss_dict["adversarial"]], 'r', label="Adversarial Loss",
             alpha=0.5)
    plt.plot(list(range(1, len(train_loss_dict["generator_total"]) + 1)), [ls / 20 for ls in train_loss_dict["generator_total"]], 'm',
             label="Total Generator Loss", alpha=0.5)
    plt.plot(list(range(1, len(train_loss_dict["discriminator_mse"]) + 1)), train_loss_dict["discriminator_mse"], 'c', label="Discriminator Loss", alpha=0.5)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Train Losses")
    plt.show()
    plt.clf()
    plt.plot(list(range(1, len(valid_loss_dict["multi_res_spectral_convergence"]) + 1)), valid_loss_dict["multi_res_spectral_convergence"], 'b',
             label="Spectral Distribution Loss", alpha=0.5)
    plt.plot(list(range(1, len(valid_loss_dict["multi_res_log_stft_mag"]) + 1)), valid_loss_dict["multi_res_log_stft_mag"], 'g',
             label="Spectral Magnitude Loss", alpha=0.5)
    plt.plot(list(range(1, len(valid_loss_dict["adversarial"]) + 1)), valid_loss_dict["adversarial"], 'r', label="Adversarial Loss", alpha=0.5)
    plt.plot(list(range(1, len(valid_loss_dict["generator_total"]) + 1)), valid_loss_dict["generator_total"], 'm', label="Total Generator Loss", alpha=0.5)
    plt.plot(list(range(1, len(valid_loss_dict["discriminator_mse"]) + 1)), valid_loss_dict["discriminator_mse"], 'c', label="Discriminator Loss", alpha=0.5)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Valid Losses")
    plt.show()


def plot_syn_training(path_to_train_val_loss_json="Models/Use/train_val_loss.json"):
    import matplotlib.pyplot as plt
    import json
    with open(path_to_train_val_loss_json, 'r') as plotting_data_file:
        train_val_loss = json.load(plotting_data_file)
    train_loss = train_val_loss[0]
    val_loss = train_val_loss[1]
    plt.plot(list(range(1, len(train_loss) + 1)), train_loss, 'b', label="Train Loss")
    plt.plot(list(range(1, len(val_loss) + 1)), val_loss, 'g', label="Valid Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def sanity_check_audio_preprocessing(path_to_wav_folder, cut_silence):
    if not path_to_wav_folder.endswith("/"):
        path_to_wav_folder = path_to_wav_folder + "/"
    path_list = [x for x in os.listdir(path_to_wav_folder) if x.endswith(".wav")]
    _, sr = sf.read(path_to_wav_folder + path_list[0])
    ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024, cut_silence=cut_silence)
    for path in path_list:
        wave, sr = sf.read(path_to_wav_folder + path)
        clean_wave = ap.normalize_audio(wave)
        print("unclean")
        sounddevice.play(wave, sr)

        print("clean")
        sounddevice.play(clean_wave, 16000)


def show_audio_lens_in_dataset(path_list):
    lens = list()
    for path in path_list:
        wave, sr = sf.read(path)
        lens.append(round(len(wave) / sr))
    lens.sort()
    plt.hist(lens)
    plt.show()


def show_all_models_params():
    from TransformerTTS.TransformerTTS import Transformer
    from FastSpeech2.FastSpeech2 import FastSpeech2
    model = Transformer(idim=133, odim=80)
    print("Number of Parameters in Transformer without Speaker Embeddings: {}".format(count_parameters(model)))
    model = Transformer(idim=133, odim=80, spk_embed_dim=256)
    print("Number of Parameters in Transformer with speedy config: {}".format(count_parameters(model)))
    model = FastSpeech2(idim=133, odim=80)
    print("Number of Parameters in FastSpeech2 without Speaker Embeddings: {}".format(count_parameters(model)))
    model = FastSpeech2(idim=133, odim=80, spk_embed_dim=256)
    print("Number of Parameters in FastSpeech2 with Speaker Embeddings: {}".format(count_parameters(model)))


if __name__ == '__main__':
    # sanity_check_audio_preprocessing("Corpora/CSS10_DE", cut_silence=True)
    # plot_fastspeech_architecture()
    # plot_transformertts_architecture()
    # plot_melgan_training()
    # plot_syn_training()
    show_att(sentence="Hallo Welt, das ist ein kurzer Satz.", lang="de", best_only=False)
