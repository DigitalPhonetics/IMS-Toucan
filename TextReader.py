import os

import torch

from InferenceInterfaces.SingleSpeakerTransformerTTSInference import SingleSpeakerTransformerTTSInference


def read_texts(lang, sentence, filename, reduction_factor=1, device="cpu"):
    tts = SingleSpeakerTransformerTTSInference(lang=lang, reduction_factor=reduction_factor, device=device)
    if type(sentence) == str:
        sentence = [sentence]
    tts.read_to_file(text_list=sentence, file_location=filename)
    del tts


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.isdir("audios"):
        os.makedirs("audios")

    read_texts(lang="de",
               sentence=["Im Frühling denkt das Rößlein, wer nicht leiden will, muss schön sein!"],
               filename="audios/test_de.wav",
               reduction_factor=1,
               device=device)

    read_texts(lang="en",
               sentence=["Lying in a field of glass, underneath the overpass.",
                         "Mangled in the shards of a metal frame.",
                         "Woken from the dream by my own name."],
               filename="audios/test_en.wav",
               reduction_factor=1,
               device=device)
