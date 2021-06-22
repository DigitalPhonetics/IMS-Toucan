import os

import torch

from InferenceInterfaces.Elizabeth_FastSpeechInference import Elizabeth_FastSpeechInference
from InferenceInterfaces.Elizabeth_TransformerTTSInference import Elizabeth_TransformerTTSInference
from InferenceInterfaces.Eva_FastSpeechInference import Eva_FastSpeechInference
from InferenceInterfaces.Eva_TransformerTTSInference import Eva_TransformerTTSInference
from InferenceInterfaces.Karlsson_FastSpeechInference import Karlsson_FastSpeechInference
from InferenceInterfaces.Karlsson_TransformerTTSInference import Karlsson_TransformerTTSInference
from InferenceInterfaces.LJSpeech_FastSpeechInference import LJSpeech_FastSpeechInference
from InferenceInterfaces.LJSpeech_TransformerTTSInference import LJSpeech_TransformerTTSInference
from InferenceInterfaces.LibriTTS_FastSpeechInference import LibriTTS_FastSpeechInference
from InferenceInterfaces.LibriTTS_TransformerTTSInference import LibriTTS_TransformerTTSInference
from InferenceInterfaces.Nancy_FastSpeechInference import Nancy_FastSpeechInference
from InferenceInterfaces.Nancy_TransformerTTSInference import Nancy_TransformerTTSInference
from InferenceInterfaces.Thorsten_FastSpeechInference import Thorsten_FastSpeechInference
from InferenceInterfaces.Thorsten_TransformerTTSInference import Thorsten_TransformerTTSInference

tts_dict = {
    "fast_thorsten"  : Thorsten_FastSpeechInference,
    "fast_lj"        : LJSpeech_FastSpeechInference,
    "fast_libri"     : LibriTTS_FastSpeechInference,
    "fast_karlsson"  : Karlsson_FastSpeechInference,
    "fast_eva"       : Eva_FastSpeechInference,
    "fast_elizabeth" : Elizabeth_FastSpeechInference,
    "fast_nancy"     : Nancy_FastSpeechInference,

    "trans_thorsten" : Thorsten_TransformerTTSInference,
    "trans_lj"       : LJSpeech_TransformerTTSInference,
    "trans_libri"    : LibriTTS_TransformerTTSInference,
    "trans_karlsson" : Karlsson_TransformerTTSInference,
    "trans_eva"      : Eva_TransformerTTSInference,
    "trans_elizabeth": Elizabeth_TransformerTTSInference,
    "trans_nancy"    : Nancy_TransformerTTSInference
    }


def read_texts(model_id, sentence, filename, device="cpu", speaker_embedding=None):
    tts = tts_dict[model_id](device=device, speaker_embedding=speaker_embedding)
    if type(sentence) == str:
        sentence = [sentence]
    tts.read_to_file(text_list=sentence, file_location=filename)
    del tts


if __name__ == '__main__':
    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.isdir("audios"):
        os.makedirs("audios")

    read_texts(model_id="trans_libri",
               sentence=["Hello world, I am a synthesis voice."],
               device=exec_device,
               speaker_embedding="glados.pt",
               filename="audios/trans_libri.wav")

    read_texts(model_id="trans_nancy",
               sentence=["Hello world, I am a synthesis voice."],
               device=exec_device,
               speaker_embedding="glados.pt",
               filename="audios/trans_nancy.wav")

    read_texts(model_id="trans_lj",
               sentence=["Hello world, I am a synthesis voice."],
               device=exec_device,
               speaker_embedding="glados.pt",
               filename="audios/trans_lj.wav")

    read_texts(model_id="fast_lj",
               sentence=["Hello world, I am a synthesis voice."],
               device=exec_device,
               speaker_embedding="glados.pt",
               filename="audios/fast_lj.wav")

    read_texts(model_id="trans_thorsten",
               sentence=["Hallo Welt, ich bin eine Synthese-Stimme."],
               device=exec_device,
               speaker_embedding="glados.pt",
               filename="audios/trans_thorsten.wav")

    read_texts(model_id="trans_karlsson",
               sentence=["Hallo Welt, ich bin eine Synthese-Stimme."],
               device=exec_device,
               speaker_embedding="glados.pt",
               filename="audios/trans_karlsson.wav")
