from FastSpeech2.FastSpeech2 import show_spectrogram as fact_spec
from InferenceInterfaces.GermanSingleSpeakerTransformerTTSInference import GermanSingleSpeakerTransformerTTSInference
from TransformerTTS.TransformerTTS import show_spectrogram as trans_spec, show_attention_plot


def show_att():
    show_attention_plot("Die drei Segel glitten wie senkrechte Papierwände über das abendglatte Wasser.")


def show_specs():
    trans_spec("Die drei Segel glitten wie senkrechte Papierwände über das abendglatte Wasser.")
    fact_spec("Die drei Segel glitten wie senkrechte Papierwände über das abendglatte Wasser.")


def read_texts():
    tts = GermanSingleSpeakerTransformerTTSInference()
    tts.read_to_file(text_list=["Hallo Welt!", "Das hier sind meine ersten zwei Sätze."], file_location="test.wav")


if __name__ == '__main__':
    read_texts()
