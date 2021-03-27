import sys

from InferenceInterfaces.SingleSpeakerTransformerTTSInference import SingleSpeakerTransformerTTSInference

if __name__ == '__main__':
    lang = input("Which language do you want? (currently supported 'en' and 'de')\n")
    tts = SingleSpeakerTransformerTTSInference(lang=lang, reduction_factor=1)
    while True:
        text = input("\nWhat should I say? (or 'exit')\n")
        if text == "exit":
            sys.exit()
        tts.read_aloud(text)
