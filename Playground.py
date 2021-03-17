"""
A place for demos, visualizations and sanity checks
"""

import os

import sounddevice
import soundfile as sf
import torch
import torchviz

from FastSpeech2.FastSpeech2 import FastSpeech2
from FastSpeech2.FastSpeech2 import show_spectrogram as fast_spec
from FastSpeech2.FastSpeechDataset import FastSpeechDataset
from InferenceInterfaces.EnglishSingleSpeakerTransformerTTSInference import EnglishSingleSpeakerTransformerTTSInference
from InferenceInterfaces.GermanSingleSpeakerTransformerTTSInference import GermanSingleSpeakerTransformerTTSInference
from PreprocessingForTTS.ProcessAudio import AudioPreprocessor
from TransformerTTS.TransformerTTS import show_spectrogram as trans_spec, show_attention_plot
from Utility.path_to_transcript_dicts import build_path_to_transcript_dict_css10de


def show_att(lang="en", best_only=False):
    if lang == "en":
        show_attention_plot(
            "Many animals of even complex structure which "
            "live parasitically within others are wholly "
            "devoid of an alimentary cavity.",
            lang=lang, best_only=best_only)
    elif lang == "de":
        show_attention_plot("Hallo Welt, ich spreche!", lang=lang, best_only=best_only)


def show_specs(lang="en"):
    if lang == "de":
        trans_spec("Hallo Welt, ich spreche!", lang=lang)
        fast_spec("Hallo Welt, ich spreche!", lang=lang)
    elif lang == "en":
        trans_spec(
            "Many animals of even complex structure which "
            "live parasitically within others are wholly "
            "devoid of an alimentary cavity.",
            lang=lang)
        fast_spec("Hello world, I am speaking!", lang=lang)


def read_texts(lang="en"):
    if lang == "de":
        tts = GermanSingleSpeakerTransformerTTSInference()
        tts.read_to_file(text_list=["Hallo Welt!", "Ich spreche."], file_location="test_de.wav")
    elif lang == "en":
        tts = EnglishSingleSpeakerTransformerTTSInference()
        tts.read_to_file(text_list=[
            "Many animals of even complex structure which "
            "live parasitically within others are wholly "
            "devoid of an alimentary cavity."],
            file_location="test_en.wav")


def plot_fastspeech_architecture():
    device = torch.device("cpu")
    path_to_transcript_dict = build_path_to_transcript_dict_css10de()
    css10_testing = FastSpeechDataset(path_to_transcript_dict,
                                      train="testing",
                                      acoustic_model_name="Transformer_German_Single.pt",
                                      loading_processes=1)
    model = FastSpeech2(idim=132, odim=80, spk_embed_dim=None).to(device)
    datapoint = css10_testing[0]
    out = model.inference(text=torch.LongTensor(datapoint[0]).squeeze(0).to(device),
                          speech=torch.Tensor(datapoint[2]).to(device),
                          durations=torch.LongTensor(datapoint[4]).to(device),
                          pitch=torch.Tensor(datapoint[5]).to(device),
                          energy=torch.Tensor(datapoint[6]).to(device),
                          spembs=None,
                          use_teacher_forcing=True)
    torchviz.make_dot(out, dict(model.named_parameters())).render("fastspeech2_graph", format="pdf")


def plot_melgan_training(path_to_train_loss_json="Models/Use/train_loss.json",
                         path_to_valid_loss_json="Models/Use/valid_loss.json"):
    import matplotlib.pyplot as plt
    import json
    with open(path_to_train_loss_json, 'r') as plotting_data_file:
        train_loss_dict = json.load(plotting_data_file)
    with open(path_to_valid_loss_json, 'r') as plotting_data_file:
        valid_loss_dict = json.load(plotting_data_file)
    plt.plot(list(range(1, len(train_loss_dict["multi_res_spectral_convergence"]) + 1)),
             train_loss_dict["multi_res_spectral_convergence"],
             'b',
             label="Spectral Distribution Loss",
             alpha=0.5)
    plt.plot(list(range(1, len(train_loss_dict["multi_res_log_stft_mag"]) + 1)),
             train_loss_dict["multi_res_log_stft_mag"],
             'g',
             label="Spectral Magnitude Loss",
             alpha=0.5)
    plt.plot(list(range(1, len(train_loss_dict["adversarial"]) + 1)),
             train_loss_dict["adversarial"],
             'r',
             label="Adversarial Loss",
             alpha=0.5)
    plt.plot(list(range(1, len(train_loss_dict["generator_total"]) + 1)),
             [ls / 20 for ls in train_loss_dict["generator_total"]],  # to balance the lambdas of stft and adv
             'm',
             label="Total Generator Loss",
             alpha=0.5)
    plt.plot(list(range(1, len(train_loss_dict["discriminator_mse"]) + 1)),
             train_loss_dict["discriminator_mse"],
             'c',
             label="Discriminator Loss",
             alpha=0.5)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Train Losses")
    plt.show()
    plt.clf()
    plt.plot(list(range(1, len(valid_loss_dict["multi_res_spectral_convergence"]) + 1)),
             valid_loss_dict["multi_res_spectral_convergence"],
             'b',
             label="Spectral Distribution Loss",
             alpha=0.5)
    plt.plot(list(range(1, len(valid_loss_dict["multi_res_log_stft_mag"]) + 1)),
             valid_loss_dict["multi_res_log_stft_mag"],
             'g',
             label="Spectral Magnitude Loss",
             alpha=0.5)
    plt.plot(list(range(1, len(valid_loss_dict["adversarial"]) + 1)),
             valid_loss_dict["adversarial"],
             'r',
             label="Adversarial Loss",
             alpha=0.5)
    plt.plot(list(range(1, len(valid_loss_dict["generator_total"]) + 1)),
             valid_loss_dict["generator_total"],
             'm',
             label="Total Generator Loss",
             alpha=0.5)
    plt.plot(list(range(1, len(valid_loss_dict["discriminator_mse"]) + 1)),
             valid_loss_dict["discriminator_mse"],
             'c',
             label="Discriminator Loss",
             alpha=0.5)
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


def sanity_check_audio_preprocessing(path_to_wav_folder):
    path_list = os.listdir(path_to_wav_folder)
    _, sr = sf.read(path_to_wav_folder + path_list[0])
    ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024)
    for path in path_list:
        wave, sr = sf.read(path_to_wav_folder + path)
        clean_wave = ap.normalize_audio(wave)
        print("unclean")
        sounddevice.play(wave, sr)
        sounddevice.wait()
        print("clean")
        sounddevice.play(clean_wave, 16000)
        sounddevice.wait()


def show_all_models_params():
    from TransformerTTS.TransformerTTS import Transformer
    from FastSpeech2.FastSpeech2 import FastSpeech2
    model = Transformer(idim=131, odim=80)
    print("Number of Parameters in Transformer without Speaker Embeddings: {}".format(count_parameters(model)))
    model = Transformer(idim=131, odim=80, spk_embed_dim=256)
    print("Number of Parameters in Transformer with speedy config: {}".format(count_parameters(model)))
    model = FastSpeech2(idim=131, odim=80)
    print("Number of Parameters in FastSpeech2 without Speaker Embeddings: {}".format(count_parameters(model)))
    model = FastSpeech2(idim=131, odim=80, spk_embed_dim=256)
    print("Number of Parameters in FastSpeech2 with Speaker Embeddings: {}".format(count_parameters(model)))


if __name__ == '__main__':
    # plot_melgan_training()
    show_att(lang="en", best_only=True)
    read_texts(lang="en")
    show_specs(lang="en")
