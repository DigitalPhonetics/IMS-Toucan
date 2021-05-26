"""
A place for demos, visualizations and sanity checks
"""

import json

import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torchviz

from FastSpeech2.FastSpeech2 import FastSpeech2
from InferenceInterfaces.Elizabeth_TransformerTTSInference import Elizabeth_TransformerTTSInference
from InferenceInterfaces.Eva_TransformerTTSInference import Eva_TransformerTTSInference
from InferenceInterfaces.Karlsson_TransformerTTSInference import Karlsson_TransformerTTSInference
from InferenceInterfaces.LJSpeech_TransformerTTSInference import LJSpeech_TransformerTTSInference
from InferenceInterfaces.LibriTTS_TransformerTTSInference import LibriTTS_TransformerTTSInference
from InferenceInterfaces.Nancy_TransformerTTSInference import Nancy_TransformerTTSInference
from InferenceInterfaces.Thorsten_TransformerTTSInference import Thorsten_TransformerTTSInference
from TransformerTTS.TransformerTTS import Transformer


def view_attention_heads(model_id, sentence):
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    tts_dict = {
        "trans_thorsten": Thorsten_TransformerTTSInference,
        "trans_lj": LJSpeech_TransformerTTSInference,
        "trans_libri": LibriTTS_TransformerTTSInference,
        "trans_karl": Karlsson_TransformerTTSInference,
        "trans_eva": Eva_TransformerTTSInference,
        "trans_elizabeth": Elizabeth_TransformerTTSInference,
        "trans_nancy": Nancy_TransformerTTSInference
    }
    tts = tts_dict[model_id](speaker_embedding="glados.pt")
    tts.plot_attentions(sentence=sentence)


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


def plot_transformertts_architecture():
    text = torch.LongTensor([1, 2, 3, 4])
    speech = torch.zeros(80, 50)
    model = Transformer(idim=166, odim=80, spk_embed_dim=None)
    out = model.inference(text=text, speech=speech, speaker_embeddings=None, use_teacher_forcing=False)
    torchviz.make_dot(out, dict(model.named_parameters())).render("transformertts_graph", format="png")


def plot_melgan_training(path_to_train_loss_json="Models/Use/train_loss.json"):
    with open(path_to_train_loss_json, 'r') as plotting_data_file:
        train_loss_dict = json.load(plotting_data_file)
    plt.plot(list(range(1, len(train_loss_dict["multi_res_spectral_convergence"]) + 1)),
             train_loss_dict["multi_res_spectral_convergence"], 'b',
             label="Spectral Distribution Loss", alpha=0.5)
    plt.plot(list(range(1, len(train_loss_dict["multi_res_log_stft_mag"]) + 1)),
             train_loss_dict["multi_res_log_stft_mag"], 'g',
             label="Spectral Magnitude Loss", alpha=0.5)
    plt.plot(list(range(1, len(train_loss_dict["adversarial"]) + 1)),
             [a / 12 for a in train_loss_dict["adversarial"]], 'r',
             label="Adversarial Loss", alpha=0.5)
    plt.plot(list(range(1, len(train_loss_dict["discriminator_mse"]) + 1)),
             train_loss_dict["discriminator_mse"], 'c',
             label="Discriminator Loss", alpha=0.5)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("MelGAN Losses")
    plt.show()


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
    from TransformerTTS.TransformerTTS import Transformer
    from FastSpeech2.FastSpeech2 import FastSpeech2
    model = Transformer(idim=166, odim=80)
    print("Number of Parameters in Transformer without Speaker Embeddings: {}".format(count_parameters(model)))
    model = Transformer(idim=166, odim=80, spk_embed_dim=256)
    print("Number of Parameters in Transformer with speedy config: {}".format(count_parameters(model)))
    model = FastSpeech2(idim=166, odim=80)
    print("Number of Parameters in FastSpeech2 without Speaker Embeddings: {}".format(count_parameters(model)))
    model = FastSpeech2(idim=166, odim=80, spk_embed_dim=256)
    print("Number of Parameters in FastSpeech2 with Speaker Embeddings: {}".format(count_parameters(model)))


if __name__ == '__main__':
    view_attention_heads("trans_libri", sentence="This is a complicated sentence, it even contains a pause!")
    view_attention_heads("trans_libri", sentence="Also maybe let's try another one.")
    view_attention_heads("trans_libri", sentence="This one, however, contains quite the significant amount of pauses, dunnit?.")
