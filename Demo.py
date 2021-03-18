import sys

from InferenceInterfaces.EnglishSingleSpeakerTransformerTTSInference import EnglishSingleSpeakerTransformerTTSInference

if __name__ == '__main__':
    tts = EnglishSingleSpeakerTransformerTTSInference()
    while True:
        text = input("\nWhat should I say? (or 'exit')\n")
        if text == "exit":
            sys.exit()
        tts.read_aloud(text)
