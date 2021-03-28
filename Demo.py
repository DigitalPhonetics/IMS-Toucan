import sys

import torch

from InferenceInterfaces.SingleSpeakerTransformerTTSInference import SingleSpeakerTransformerTTSInference

if __name__ == '__main__':
    lang = input("Which language do you want? (currently supported 'en' and 'de')\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = SingleSpeakerTransformerTTSInference(lang=lang, reduction_factor=1, device=device)
    while True:
        text = input("\nWhat should I say? (or 'exit')\n")
        if text == "exit":
            sys.exit()
        tts.read_aloud(text)
