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

    tl = """And just cause he's gone, it doesn't change the fact:
    He was a bastard in life, thus a bastard in death.""".split("\n")

    read_texts(lang="en",
               sentence=tl,
               filename="styrofoam_plates.wav",
               reduction_factor=1,
               device=device)

    read_texts(lang="en",
               sentence=[
                   "Lying in a field of glass, underneath the overpass.",
                   "Mangled in the shards of a metal frame.",
                   "Woken from the dream, by my own name."],
               filename="test_en.wav",
               reduction_factor=1,
               device=device)

    read_texts(lang="de",
               sentence=["Hallo Welt!"],
               filename="test_de.wav",
               reduction_factor=1,
               device=device)
