import sys

import torch

from InferenceInterfaces.CSS10_DE_FastSpeechInference import CSS10_DE_FastSpeechInference
from InferenceInterfaces.CSS10_DE_TransformerTTSInference import CSS10_DE_TransformerTTSInference
from InferenceInterfaces.LJSpeech_FastSpeechInference import LJSpeech_FastSpeechInference
from InferenceInterfaces.LJSpeech_TransformerTTSInference import LJSpeech_TransformerTTSInference
from InferenceInterfaces.LibriTTS_FastSpeechInference import LibriTTS_FastSpeechInference
from InferenceInterfaces.LibriTTS_TransformerTTSInference import LibriTTS_TransformerTTSInference
from InferenceInterfaces.Thorsten_FastSpeechInference import Thorsten_FastSpeechInference
from InferenceInterfaces.Thorsten_TransformerTTSInference import Thorsten_TransformerTTSInference

tts_dict = {
    "fast_thorsten" : Thorsten_FastSpeechInference,
    "fast_lj"       : LJSpeech_FastSpeechInference,
    "fast_css10_de" : CSS10_DE_FastSpeechInference,
    "fast_libri"    : LibriTTS_FastSpeechInference,

    "trans_thorsten": Thorsten_TransformerTTSInference,
    "trans_lj"      : LJSpeech_TransformerTTSInference,
    "trans_css10_de": CSS10_DE_TransformerTTSInference,
    "trans_libri"   : LibriTTS_TransformerTTSInference

    }

if __name__ == '__main__':
    model_id = input("Which model do you want? \nCurrently supported are: {}\n".format("".join("\n\t- {}".format(key) for key in tts_dict.keys())))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = tts_dict[model_id](device=device)
    while True:
        text = input("\nWhat should I say? (or 'exit')\n")
        if text == "exit":
            sys.exit()
        tts.read_aloud(text, view=True)
