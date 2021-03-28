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

    tl = """There's a saltwater film on the jar of your ashes:
    I threw them to sea, but a gust blew them backwards. 
    And the sting in my eyes, that you then inflicted, was par for the course, just as when you were living.
    It's no stretch to say, you were not quite a father.
    But a donor of seeds, to a poor single mother, that would raise us alone.
    We never saw the money, that went down your throat, through the hole in your belly.
    Thirteen years old in the suburbs of Denver, standing in line for Thanksgiving dinner, at the catholic church.
    The servers wore crosses, to shield from the sufferance, plaguing the others.
    Styrofoam plates, cafeteria tables, charity reeks of cheap wine and pity, and I'm thinking of you. 
    I do every year, as we count all our blessings and wonder what we're doing here.
    You're a disgrace to the concept of family. 
    The priest won't divulge that fact in his homily, and I'll stand up and scream, if the mourning remain quiet.
    You can deck out a lie in a suit, but I won't buy it.
    I won't join the procession that's speaking their piece, using five dollar words while praising his integrity.
    And just cause he's gone, it doesn't change the fact:
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
