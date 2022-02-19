import os
import sys
import warnings

import torch

from InferenceInterfaces.InferenceMultiSpeakerMultiLingualFastSpeech2 import InferenceMultiSpeakerMultiLingualFastSpeech2

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    available_models = os.listdir("Models")
    available_fastspeech_models = list()
    for model in available_models:
        if model.startswith("FastSpeech2_"):
            available_fastspeech_models.append(model.lstrip("FastSpeech_2"))
    model_id = input("Which model do you want? \nCurrently supported are: {}\n".format("".join("\n\t- {}".format(key) for key in available_fastspeech_models)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = InferenceMultiSpeakerMultiLingualFastSpeech2(device=device, model_name=model_id)
    while True:
        text = input("\nWhat should I say? (or 'exit')\n")
        if text == "exit":
            sys.exit()
        tts.read_aloud(text, view=True, blocking=False)
