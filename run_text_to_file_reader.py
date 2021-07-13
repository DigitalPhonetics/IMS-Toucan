import os

import torch

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
    "fast_nancy"     : Nancy_FastSpeechInference,

    "trans_thorsten" : Thorsten_TransformerTTSInference,
    "trans_lj"       : LJSpeech_TransformerTTSInference,
    "trans_libri"    : LibriTTS_TransformerTTSInference,
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

    read_texts(model_id="fast_libri",
               sentence=["Hello world, I am a synthesis voice."],
               device=exec_device,
               speaker_embedding="glados.pt",
               filename="audios/fast_libri.wav")

    read_texts(model_id="fast_nancy",
               sentence=["Hello world, I am a synthesis voice."],
               device=exec_device,
               speaker_embedding="glados.pt",
               filename="audios/fast_nancy.wav")

    read_texts(model_id="fast_lj",
               sentence=["Hello world, I am a synthesis voice."],
               device=exec_device,
               speaker_embedding="glados.pt",
               filename="audios/fast_lj.wav")

    read_texts(model_id="fast_thorsten",
               sentence=["Hallo Welt, ich bin eine Synthese-Stimme."],
               device=exec_device,
               speaker_embedding="glados.pt",
               filename="audios/fast_thorsten.wav")

    read_texts(model_id="trans_lj",
               sentence=["Betty Botter bought some butter, but she said the butter's bitter."],
               device=exec_device,
               speaker_embedding="glados.pt",
               filename="audios/trans_lj.wav")
