import torch
import torchviz

from FastSpeech2.FastSpeech2 import FastSpeech2
from FastSpeech2.FastSpeech2 import show_spectrogram as fact_spec
from FastSpeech2.FastSpeechDataset import FastSpeechDataset
from InferenceInterfaces.GermanSingleSpeakerTransformerTTSInference import GermanSingleSpeakerTransformerTTSInference
from PipelineFastSpeech2_CSS10 import build_path_to_transcript_dict
from TransformerTTS.TransformerTTS import show_spectrogram as trans_spec, show_attention_plot


def show_att():
    show_attention_plot("Die drei Segel glitten wie senkrechte Papierwände über das abendglatte Wasser.")


def show_specs():
    trans_spec("Die drei Segel glitten wie senkrechte Papierwände über das abendglatte Wasser.")
    fact_spec("Die drei Segel glitten wie senkrechte Papierwände über das abendglatte Wasser.")


def read_texts():
    tts = GermanSingleSpeakerTransformerTTSInference()
    tts.read_to_file(text_list=["Hallo Welt!", "Das hier sind meine ersten zwei Sätze."], file_location="test.wav")


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
    print(torch.LongTensor(datapoint[0]).to(device).shape, torch.Tensor(datapoint[2]).to(device).shape,
          torch.Tensor(datapoint[4]).to(device).shape, torch.Tensor(datapoint[5]).to(device).shape,
          torch.Tensor(datapoint[6]).to(device).shape)
    out = model.inference(text=torch.LongTensor(datapoint[0]).to(device),
                          speech=torch.Tensor(datapoint[2]).to(device),
                          durations=torch.Tensor(datapoint[4]).to(device),
                          pitch=torch.Tensor(datapoint[5]).to(device),
                          energy=torch.Tensor(datapoint[6]).to(device),
                          spembs=None,
                          use_teacher_forcing=True)
    torchviz.make_dot(out, dict(model.named_parameters())).render("fastspeech2_graph", format="pdf")


if __name__ == '__main__':
    plot_fastspeech_architecture()
