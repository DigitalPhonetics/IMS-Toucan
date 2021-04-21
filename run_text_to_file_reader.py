import os

import torch

from InferenceInterfaces.Elizabeth_FastSpeechInference import Elizabeth_FastSpeechInference
from InferenceInterfaces.Elizabeth_TransformerTTSInference import Elizabeth_TransformerTTSInference
from InferenceInterfaces.Eva_FastSpeechInference import Eva_FastSpeechInference
from InferenceInterfaces.Eva_TransformerTTSInference import Eva_TransformerTTSInference
from InferenceInterfaces.Karlsson_FastSpeechInference import Karlsson_FastSpeechInference
from InferenceInterfaces.Karlsson_TransformerTTSInference import Karlsson_TransformerTTSInference
from InferenceInterfaces.LJSpeech_FastSpeechInference import LJSpeech_FastSpeechInference
from InferenceInterfaces.LJSpeech_FastSpeechInference_new_hyp import LJSpeech_FastSpeechInference_new_hyp
from InferenceInterfaces.LJSpeech_TransformerTTSInference import LJSpeech_TransformerTTSInference
from InferenceInterfaces.LibriTTS_FastSpeechInference import LibriTTS_FastSpeechInference
from InferenceInterfaces.LibriTTS_TransformerTTSInference import LibriTTS_TransformerTTSInference
from InferenceInterfaces.Thorsten_FastSpeechInference import Thorsten_FastSpeechInference
from InferenceInterfaces.Thorsten_TransformerTTSInference import Thorsten_TransformerTTSInference

tts_dict = {
    "fast_thorsten"  : Thorsten_FastSpeechInference,
    "fast_lj"        : LJSpeech_FastSpeechInference,
    "fast_libri"     : LibriTTS_FastSpeechInference,
    "fast_karl"      : Karlsson_FastSpeechInference,
    "fast_eva"       : Eva_FastSpeechInference,
    "fast_elizabeth" : Elizabeth_FastSpeechInference,
    "fast_lj_new_hyp": LJSpeech_FastSpeechInference_new_hyp,

    "trans_thorsten" : Thorsten_TransformerTTSInference,
    "trans_lj"       : LJSpeech_TransformerTTSInference,
    "trans_libri"    : LibriTTS_TransformerTTSInference,
    "trans_karl"     : Karlsson_TransformerTTSInference,
    "trans_eva"      : Eva_TransformerTTSInference,
    "trans_elizabeth": Elizabeth_TransformerTTSInference
    }


def read_texts(model_id, sentence, filename, device="cpu", speaker_embedding=None):
    tts = tts_dict[model_id](device=device, speaker_embedding=speaker_embedding)
    if type(sentence) == str:
        sentence = [sentence]
    tts.read_to_file(text_list=sentence, file_location=filename)
    del tts


if __name__ == '__main__':
    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.isdir("audios"):
        os.makedirs("audios")

    read_texts(model_id="fast_libri",
               sentence=["Those of you who volunteered to be injected with praying mantis DNA, I've got some good news and some bad news.",
                         "Bad news is we're postponing those tests indefinitely.",
                         "Good news is we've got a much better test for you: fighting an army of mantis men.",
                         "Pick up a rifle and follow the yellow line.",
                         "You'll know when the test starts."],
               filename="audios/turret.wav",
               device=exec_device,
               speaker_embedding="turret.pt")

    read_texts(model_id="fast_libri",
               sentence=["All right, I've been thinking.",
                         "When life gives you lemons?",
                         "Don't make lemonade.",
                         "Make life take the lemons back!",
                         "Get mad!",
                         "I don't want your damn lemons!",
                         "What am I supposed to do with these?"],
               filename="audios/cave_lemons.wav",
               device=exec_device,
               speaker_embedding="cave_johnson.pt")

    read_texts(model_id="fast_libri",
               sentence=["Okay.",
                         "Look.",
                         "We both said a lot of things that you're going to regret.",
                         "But I think we can put our differences behind us.",
                         "For science.",
                         "You monster!"],
               filename="audios/glados_regret.wav",
               device=exec_device,
               speaker_embedding="glados.pt")

    read_texts(model_id="fast_lj", sentence="""Peter Piper picked a peck of pickled peppers.
A peck of pickled peppers Peter Piper picked.
If Peter Piper picked a peck of pickled peppers, where’s the peck of pickled peppers Peter Piper picked?
Betty Botter bought some butter, but she said the butter’s bitter.
If I put it in my batter, it will make my batter bitter!
But a bit of better butter will make my batter better.
So ‘twas better Betty Botter bought a bit of better butter.
How much wood would a woodchuck chuck if a woodchuck could chuck wood?
He would chuck, he would, as much as he could, and chuck as much wood, as a woodchuck would if a woodchuck could chuck wood.
She sells seashells by the seashore.
How can a clam cram in a clean cream can?
I scream, you scream, we all scream for ice cream!
Susie works in a shoeshine shop. Where she shines she sits, and where she sits she shines.
Fuzzy Wuzzy was a bear. Fuzzy Wuzzy had no hair. Fuzzy Wuzzy wasn’t fuzzy, was he?
Can you can a can as a canner can can a can?
I have got a date at a quarter to eight; I’ll see you at the gate, so don’t be late.
You know New York, you need New York, you know you need unique New York
I saw a kitten eating chicken in the kitchen.
If a dog chews shoes, whose shoes does he choose?
I thought I thought of thinking of thanking you.
I wish to wash my Irish wristwatch.
Near an ear, a nearer ear, a nearly eerie ear.
Nine nice night nurses nursing nicely.
Wayne went to wales to watch walruses.""".split("\n"), filename="audios/fast_lj.wav", device=exec_device)

    read_texts(model_id="fast_lj_new_hyp", sentence="""Peter Piper picked a peck of pickled peppers.
A peck of pickled peppers Peter Piper picked.
If Peter Piper picked a peck of pickled peppers, where’s the peck of pickled peppers Peter Piper picked?
Betty Botter bought some butter, but she said the butter’s bitter.
If I put it in my batter, it will make my batter bitter!
But a bit of better butter will make my batter better.
So ‘twas better Betty Botter bought a bit of better butter.
How much wood would a woodchuck chuck if a woodchuck could chuck wood?
He would chuck, he would, as much as he could, and chuck as much wood, as a woodchuck would if a woodchuck could chuck wood.
She sells seashells by the seashore.
How can a clam cram in a clean cream can?
I scream, you scream, we all scream for ice cream!
Susie works in a shoeshine shop. Where she shines she sits, and where she sits she shines.
Fuzzy Wuzzy was a bear. Fuzzy Wuzzy had no hair. Fuzzy Wuzzy wasn’t fuzzy, was he?
Can you can a can as a canner can can a can?
I have got a date at a quarter to eight; I’ll see you at the gate, so don’t be late.
You know New York, you need New York, you know you need unique New York
I saw a kitten eating chicken in the kitchen.
If a dog chews shoes, whose shoes does he choose?
I thought I thought of thinking of thanking you.
I wish to wash my Irish wristwatch.
Near an ear, a nearer ear, a nearly eerie ear.
Nine nice night nurses nursing nicely.
Wayne went to wales to watch walruses.""".split("\n"), filename="audios/fast_lj_new_hyp.wav", device=exec_device)

    read_texts(model_id="fast_thorsten", sentence=["Hallo, ich bin eine deutsche Stimme."], filename="audios/fast_thorsten.wav", device=exec_device)
