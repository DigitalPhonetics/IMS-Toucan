import sys
import warnings

import torch

from InferenceInterfaces.Eva_FastSpeech2 import Eva_FastSpeech2
from InferenceInterfaces.Eva_Tacotron2 import Eva_Tacotron2
from InferenceInterfaces.HokusPokus_FastSpeech2 import HokusPokus_FastSpeech2
from InferenceInterfaces.HokusPokus_Tacotron2 import HokusPokus_Tacotron2
from InferenceInterfaces.Karlsson_FastSpeech2 import Karlsson_FastSpeech2
from InferenceInterfaces.Karlsson_Tacotron2 import Karlsson_Tacotron2
from InferenceInterfaces.LowRes_FastSpeech2 import LowRes_FastSpeech2 as fast_low
from InferenceInterfaces.LowRes_Tacotron2 import LowRes_Tacotron2 as taco_low
from InferenceInterfaces.Nancy_FastSpeech2 import Nancy_FastSpeech2
from InferenceInterfaces.Nancy_Tacotron2 import Nancy_Tacotron2

tts_dict = {
    "fast_nancy"   : Nancy_FastSpeech2,
    "fast_hokus"   : HokusPokus_FastSpeech2,

    "taco_nancy"   : Nancy_Tacotron2,
    "taco_hokus"   : HokusPokus_Tacotron2,

    "taco_low"     : taco_low,
    "fast_low"     : fast_low,

    "taco_eva"     : Eva_Tacotron2,
    "fast_eva"     : Eva_FastSpeech2,

    "taco_karlsson": Karlsson_Tacotron2,
    "fast_karlsson": Karlsson_FastSpeech2,
    }

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    model_id = input("Which model do you want? \nCurrently supported are: {}\n".format("".join("\n\t- {}".format(key) for key in tts_dict.keys())))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = tts_dict[model_id](device=device)
    while True:
        text = input("\nWhat should I say? (or 'exit')\n")
        if text == "exit":
            sys.exit()
        tts.read_aloud(text, view=True, blocking=False)
