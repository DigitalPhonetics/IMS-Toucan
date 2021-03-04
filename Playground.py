import torch
import torchviz

from FastSpeech2.FastSpeech2 import FastSpeech2
from FastSpeech2.FastSpeech2 import show_spectrogram as fast_spec
from FastSpeech2.FastSpeechDataset import FastSpeechDataset
from InferenceInterfaces.EnglishSingleSpeakerTransformerTTSInference import EnglishSingleSpeakerTransformerTTSInference
from InferenceInterfaces.GermanSingleSpeakerTransformerTTSInference import GermanSingleSpeakerTransformerTTSInference
from Pipeline_FastSpeech2_CSS10DE import build_path_to_transcript_dict
from TransformerTTS.TransformerTTS import show_spectrogram as trans_spec, show_attention_plot


def show_att(lang="en"):
    if lang == "en":
        show_attention_plot("Hello world, I am speaking!", lang=lang)
    elif lang == "de":
        show_attention_plot("Hallo Welt, ich spreche!", lang=lang)


def show_specs(lang="en"):
    if lang == "de":
        trans_spec("Hallo Welt, ich spreche!", lang=lang)
        fast_spec("Hallo Welt, ich spreche!", lang=lang)
    elif lang == "en":
        trans_spec("Hello world, I am speaking!", lang=lang)
        fast_spec("Hello world, I am speaking!", lang=lang)


def read_texts(lang="en"):
    if lang == "de":
        tts = GermanSingleSpeakerTransformerTTSInference()
        tts.read_to_file(text_list=["Hallo Welt!", "Ich spreche."], file_location="test_de.wav")
    elif lang == "en":
        tts = EnglishSingleSpeakerTransformerTTSInference()
        tts.read_to_file(text_list=["Hello world!", "I am speaking."], file_location="test_en.wav")


def plot_fastspeech_architecture():
    device = torch.device("cpu")
    path_to_transcript_dict = build_path_to_transcript_dict()
    css10_testing = FastSpeechDataset(path_to_transcript_dict,
                                      train="testing",
                                      acoustic_model_name="Transformer_German_Single.pt",
                                      save=False,
                                      load=False,
                                      loading_processes=1)
    model = FastSpeech2(idim=132, odim=80, spk_embed_dim=None).to(device)
    datapoint = css10_testing[0]
    out = model.inference(text=torch.LongTensor(datapoint[0]).to(device),
                          speech=torch.Tensor(datapoint[2]).to(device),
                          durations=torch.LongTensor(datapoint[4]).to(device),
                          pitch=torch.Tensor(datapoint[5]).to(device),
                          energy=torch.Tensor(datapoint[6]).to(device),
                          spembs=None,
                          use_teacher_forcing=True)
    torchviz.make_dot(out, dict(model.named_parameters())).render("fastspeech2_graph", format="pdf")


def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


if __name__ == '__main__':
    show_att(lang="en")
    show_att(lang="de")
