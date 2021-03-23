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
from FastSpeech2.FastSpeech2 import show_spectrogram as fast_spec
from InferenceInterfaces.SingleSpeakerTransformerTTSInference import SingleSpeakerTransformerTTSInference
from MelGAN.MelGANGenerator import MelGANGenerator
from PreprocessingForTTS.ProcessAudio import AudioPreprocessor
from TransformerTTS.TransformerTTS import Transformer
from TransformerTTS.TransformerTTS import show_spectrogram as trans_spec, show_attention_plot


def show_att(lang="en", best_only=False, teacher_forcing=False):
    if lang == "en":
        show_attention_plot(
            "Many animals of even complex structure which "
            "live parasitically within others are wholly "
            "devoid of an alimentary cavity.",
            lang=lang, best_only=best_only, teacher_forcing=teacher_forcing)
    elif lang == "de":
        show_attention_plot("Es war einmal – welcher Autor "
                            "darf es jetzt wohl noch wagen, "
                            "sein Geschichtlein also zu beginnen.", lang=lang, best_only=best_only,
                            teacher_forcing=teacher_forcing)


def show_specs(lang="en"):
    if lang == "de":
        trans_spec("Es war einmal – welcher Autor darf es "
                   "jetzt wohl noch wagen, sein Geschichtlein "
                   "also zu beginnen.", lang=lang)
        fast_spec("Es war einmal – welcher Autor darf es "
                  "jetzt wohl noch wagen, sein Geschichtlein "
                  "also zu beginnen.", lang=lang)
    elif lang == "en":
        trans_spec(
            "Many animals of even complex structure which "
            "live parasitically within others are wholly "
            "devoid of an alimentary cavity.",
            lang=lang)
        fast_spec("Many animals of even complex structure which "
                  "live parasitically within others are wholly "
                  "devoid of an alimentary cavity.", lang=lang)


def read_texts(lang="en", sentence=None):
    tts = SingleSpeakerTransformerTTSInference(lang=lang)
    if lang == "de":
        if sentence is None:
            tts.read_to_file(text_list=["Es war einmal – welcher "
                                        "Autor darf es jetzt wohl "
                                        "noch wagen, sein Geschichtlein "
                                        "also zu beginnen."], file_location="test_de.wav")
        else:
            tts.read_to_file(text_list=[sentence], file_location="test_de.wav")
    elif lang == "en":
        if sentence is None:
            tts.read_to_file(text_list=[
                "Many animals of even complex structure which "
                "live parasitically within others are wholly "
                "devoid of an alimentary cavity."],
                file_location="test_en.wav")
        else:
            tts.read_to_file(text_list=[sentence], file_location="test_en.wav")


def plot_fastspeech_architecture():
    text = torch.LongTensor([1, 2, 3, 4])
    speech = torch.zeros(80, 50)
    durations = torch.LongTensor([1, 2, 3, 4])
    pitch = torch.Tensor([1.0]).unsqueeze(0)
    energy = torch.Tensor([1.0]).unsqueeze(0)
    model = FastSpeech2(idim=134, odim=80, spk_embed_dim=None)
    out = model.inference(text=text,
                          speech=speech,
                          durations=durations,
                          pitch=pitch,
                          energy=energy,
                          spembs=None,
                          use_teacher_forcing=True)
    torchviz.make_dot(out, dict(model.named_parameters())).render("fastspeech2_graph", format="pdf")


def plot_transformertts_architecture():
    text = torch.LongTensor([1, 2, 3, 4])
    speech = torch.zeros(80, 50)
    model = Transformer(idim=134, odim=80, spk_embed_dim=None)
    out = model.inference(text=text,
                          speech=speech,
                          spembs=None,
                          use_teacher_forcing=False)
    torchviz.make_dot(out, dict(model.named_parameters())).render("transformertts_graph", format="png")


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


def sanity_check_audio_preprocessing(path_to_wav_folder, cut_silence):
    if not path_to_wav_folder.endswith("/"):
        path_to_wav_folder = path_to_wav_folder + "/"
    path_list = [x for x in os.listdir(path_to_wav_folder) if x.endswith(".wav")]
    _, sr = sf.read(path_to_wav_folder + path_list[0])
    ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024,
                           cut_silence=cut_silence)
    for path in path_list:
        wave, sr = sf.read(path_to_wav_folder + path)
        clean_wave = ap.normalize_audio(wave)
        print("unclean")
        sounddevice.play(wave, sr)
        sounddevice.wait()
        print("clean")
        sounddevice.play(clean_wave, 16000)
        sounddevice.wait()


def test_spectrogram_inversion(path_to_wav="Corpora/test.wav"):
    wave, sr = sf.read(path_to_wav)
    ap = AudioPreprocessor(input_sr=sr, output_sr=16000)
    clean_wave = ap.normalize_audio(wave)
    spec = ap.audio_to_mel_spec_tensor(clean_wave, normalize=False)
    spectrogram_inverter = MelGANGenerator()
    spectrogram_inverter.load_state_dict(
        torch.load(os.path.join("Models", "Use", "MelGAN.pt"), map_location='cpu')["generator"])
    reconstructed_wave = spectrogram_inverter.inference(spec.unsqueeze(0)).squeeze(0).squeeze(0)
    import matplotlib.pyplot as plt
    import librosa.display as lbd
    fig, axes = plt.subplots(nrows=2, ncols=2)
    axes[0][0].plot(clean_wave.detach().numpy())
    axes[1][0].plot(reconstructed_wave.detach().numpy())
    lbd.specshow(spec.detach().numpy(),
                 ax=axes[0][1], sr=16000, cmap='GnBu', y_axis='mel', x_axis='time', hop_length=256)
    lbd.specshow(ap.audio_to_mel_spec_tensor(reconstructed_wave.detach().numpy(), normalize=False).detach().numpy(),
                 ax=axes[1][1], sr=16000, cmap='GnBu', y_axis='mel', x_axis='time', hop_length=256)
    axes[0][0].xaxis.set_visible(False)
    axes[0][0].yaxis.set_visible(False)
    axes[0][1].xaxis.set_visible(False)
    axes[0][1].yaxis.set_visible(False)
    axes[1][0].xaxis.set_visible(False)
    axes[1][0].yaxis.set_visible(False)
    axes[1][1].xaxis.set_visible(False)
    axes[1][1].yaxis.set_visible(False)
    axes[0][0].set_title("Original Wave")
    axes[1][0].set_title("Reconstructed Wave")
    axes[0][1].set_title("Original Spectrogram")
    axes[1][1].set_title("Reconstructed Spectrogram")
    plt.subplots_adjust(left=0.02, bottom=0.02, right=.98, top=.9, wspace=0, hspace=0.2)
    plt.show()
    sf.write("audio_orig.wav", data=clean_wave.detach().numpy(), samplerate=16000)
    sf.write("audio_reconstructed.wav", data=reconstructed_wave.detach().numpy(), samplerate=16000)


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
    model = Transformer(idim=134, odim=80)
    print("Number of Parameters in Transformer without Speaker Embeddings: {}".format(count_parameters(model)))
    model = Transformer(idim=134, odim=80, spk_embed_dim=256)
    print("Number of Parameters in Transformer with speedy config: {}".format(count_parameters(model)))
    model = FastSpeech2(idim=134, odim=80)
    print("Number of Parameters in FastSpeech2 without Speaker Embeddings: {}".format(count_parameters(model)))
    model = FastSpeech2(idim=134, odim=80, spk_embed_dim=256)
    print("Number of Parameters in FastSpeech2 with Speaker Embeddings: {}".format(count_parameters(model)))


if __name__ == '__main__':
    # sanity_check_audio_preprocessing("Corpora/CSS10_DE", cut_silence=True)
    # plot_fastspeech_architecture()
    # plot_transformertts_architecture()
    # plot_melgan_training()
    test_spectrogram_inversion()
    # show_att(lang="en", best_only=True, teacher_forcing=True)
    read_texts(lang="en",
               sentence="I am fairly good at producing unseen sentences now, but I still struggle with knowing when to stop.")
    read_texts(lang="de",
               sentence="Deutsch klingt noch sehr schlecht, ich glaube der LibriVox Hokuspokus Korpus ist ein bisschen unsauber.")
    # show_specs(lang="en")
    # show_specs(lang="de")
