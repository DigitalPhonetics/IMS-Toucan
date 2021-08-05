import sys

import torch
import warnings

from InferenceInterfaces.LJSpeech_FastSpeech import LJSpeech_FastSpeechInference
from InferenceInterfaces.LJSpeech_TransformerTTS import LJSpeech_TransformerTTSInference
from InferenceInterfaces.LibriTTS_FastSpeech import LibriTTS_FastSpeechInference
from InferenceInterfaces.LibriTTS_TransformerTTS import LibriTTS_TransformerTTSInference
from InferenceInterfaces.Nancy_FastSpeech import Nancy_FastSpeechInference
from InferenceInterfaces.Nancy_TransformerTTS import Nancy_TransformerTTSInference
from InferenceInterfaces.Thorsten_FastSpeech import Thorsten_FastSpeechInference
from InferenceInterfaces.Thorsten_TransformerTTS import Thorsten_TransformerTTSInference

tts_dict = {
    "fast_thorsten" : Thorsten_FastSpeechInference,
    "fast_lj"       : LJSpeech_FastSpeechInference,
    "fast_libri"    : LibriTTS_FastSpeechInference,
    "fast_nancy"    : Nancy_FastSpeechInference,

    "trans_thorsten": Thorsten_TransformerTTSInference,
    "trans_lj"      : LJSpeech_TransformerTTSInference,
    "trans_libri"   : LibriTTS_TransformerTTSInference,
    "trans_nancy"   : Nancy_TransformerTTSInference
    }

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    model_id = input("Which model do you want? \nCurrently supported are: {}\n".format("".join("\n\t- {}".format(key) for key in tts_dict.keys())))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = tts_dict[model_id](device=device, speaker_embedding="default_speaker_embedding.pt")
    while True:
        text = input("\nWhat should I say? (or 'exit')\n")
        if text == "exit":
            sys.exit()
        tts.read_aloud(text, view=True, blocking=False)
