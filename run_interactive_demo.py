import sys
import warnings

import torch

from InferenceInterfaces.MultiEnglish_FastSpeech2 import MultiEnglish_FastSpeech2
from InferenceInterfaces.MultiEnglish_Tacotron2 import MultiEnglish_Tacotron2
from InferenceInterfaces.Nancy_FastSpeech2 import Nancy_FastSpeech2
from InferenceInterfaces.Nancy_Tacotron2 import Nancy_Tacotron2

tts_dict = {
    "fast_multi"   : MultiEnglish_FastSpeech2,
    "fast_nancy"   : Nancy_FastSpeech2,

    "taco_multi"   : MultiEnglish_Tacotron2,
    "taco_nancy"   : Nancy_Tacotron2
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
