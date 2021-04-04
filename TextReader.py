import os

import torch

from InferenceInterfaces.CSS10_DE_FastSpeechInference import CSS10_DE_FastSpeechInference
from InferenceInterfaces.CSS10_DE_TransformerTTSInference import CSS10_DE_TransformerTTSInference
from InferenceInterfaces.LJSpeech_FastSpeechInference import LJSpeech_FastSpeechInference
from InferenceInterfaces.LJSpeech_TransformerTTSInference import LJSpeech_TransformerTTSInference
from InferenceInterfaces.LibriTTS_FastSpeechInference import LibriTTS_FastSpeechInference
from InferenceInterfaces.LibriTTS_TransformerTTSInference import LibriTTS_TransformerTTSInference
from InferenceInterfaces.Thorsten_FastSpeechInference import Thorsten_FastSpeechInference
from InferenceInterfaces.Thorsten_TransformerTTSInference import Thorsten_TransformerTTSInference

tts_dict = {
    "fast_thorsten": Thorsten_FastSpeechInference,
    "fast_lj": LJSpeech_FastSpeechInference,
    "fast_css10_de": CSS10_DE_FastSpeechInference,
    "fast_libri": LibriTTS_FastSpeechInference,

    "trans_thorsten": Thorsten_TransformerTTSInference,
    "trans_lj": LJSpeech_TransformerTTSInference,
    "trans_css10_de": CSS10_DE_TransformerTTSInference,
    "trans_libri": LibriTTS_TransformerTTSInference

}


def read_texts(model_id, sentence, filename, device="cpu"):
    tts = tts_dict[model_id](device=device)
    if type(sentence) == str:
        sentence = [sentence]
    tts.read_to_file(text_list=sentence, file_location=filename)
    del tts


tl = """Peter Piper picked a peck of pickled peppers
A peck of pickled peppers Peter Piper picked
If Peter Piper picked a peck of pickled peppers
Where’s the peck of pickled peppers Peter Piper picked?
Betty Botter bought some butter
But she said the butter’s bitter
If I put it in my batter, it will make my batter bitter
But a bit of better butter will make my batter better
So ‘twas better Betty Botter bought a bit of better butter
How much wood would a woodchuck chuck if a woodchuck could chuck wood?
He would chuck, he would, as much as he could, and chuck as much wood
As a woodchuck would if a woodchuck could chuck wood
She sells seashells by the seashore
How can a clam cram in a clean cream can?
I scream, you scream, we all scream for ice cream
I saw Susie sitting in a shoeshine shop
Susie works in a shoeshine shop. Where she shines she sits, and where she sits she shines
Fuzzy Wuzzy was a bear. Fuzzy Wuzzy had no hair. Fuzzy Wuzzy wasn’t fuzzy, was he?
Can you can a can as a canner can can a can?
I have got a date at a quarter to eight; I’ll see you at the gate, so don’t be late
You know New York, you need New York, you know you need unique New York
I saw a kitten eating chicken in the kitchen
If a dog chews shoes, whose shoes does he choose?
I thought I thought of thinking of thanking you
I wish to wash my Irish wristwatch
Near an ear, a nearer ear, a nearly eerie ear
Eddie edited it
Willie’s really weary
A big black bear sat on a big black rug
Tom threw Tim three thumbtacks
He threw three free throws
Nine nice night nurses nursing nicely
So, this is the sushi chef
Four fine fresh fish for you
Wayne went to wales to watch walruses""".split("\n")

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.isdir("audios"):
        os.makedirs("audios")

    read_texts(model_id="fast_lj",
               sentence=tl,
               filename="audios/fast_lj.wav",
               device=device)
