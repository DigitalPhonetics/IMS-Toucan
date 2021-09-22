import os

import torch

from InferenceInterfaces.LJSpeech_FastSpeech2 import LJSpeech_FastSpeech2
from InferenceInterfaces.LJSpeech_Tacotron2 import LJSpeech_Tacotron2
from InferenceInterfaces.MultiEnglish_FastSpeech2 import LibriTTS_FastSpeech2
from InferenceInterfaces.MultiEnglish_Tacotron2 import LibriTTS_Tacotron2
from InferenceInterfaces.Nancy_FastSpeech2 import Nancy_FastSpeech2
from InferenceInterfaces.Nancy_Tacotron2 import Nancy_Tacotron2
from InferenceInterfaces.Thorsten_FastSpeech2 import Thorsten_FastSpeech2
from InferenceInterfaces.Thorsten_Tacotron2 import Thorsten_Tacotron2

tts_dict = {
    "fast_thorsten": Thorsten_FastSpeech2,
    "fast_lj": LJSpeech_FastSpeech2,
    "fast_libri": LibriTTS_FastSpeech2,
    "fast_nancy": Nancy_FastSpeech2,

    "taco_thorsten": Thorsten_Tacotron2,
    "taco_lj": LJSpeech_Tacotron2,
    "taco_libri": LibriTTS_Tacotron2,
    "taco_nancy": Nancy_Tacotron2
}


def read_texts(model_id, sentence, filename, device="cpu", speaker_embedding=None):
    tts = tts_dict[model_id](device=device, speaker_embedding=speaker_embedding)
    if type(sentence) == str:
        sentence = [sentence]
    tts.read_to_file(text_list=sentence, file_location=filename)
    del tts


def save_weights(model_id):
    tts_dict[model_id](device="cpu", speaker_embedding="default_speaker_embedding.pt").save_pretrained_weights()


def read_harvard_sentences(model_id, device):
    tts = tts_dict[model_id](device=device, speaker_embedding="default_speaker_embedding.pt")

    with open("Utility/test_sentences_combined_3.txt", "r", encoding="utf8") as f:
        sents = f.read().split("\n")
    output_dir = "audios/harvard_03_{}".format(model_id)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for index, sent in enumerate(sents):
        tts.read_to_file(text_list=[sent], file_location=output_dir + "/{}.wav".format(index))

    with open("Utility/test_sentences_combined_6.txt", "r", encoding="utf8") as f:
        sents = f.read().split("\n")
    output_dir = "audios/harvard_06_{}".format(model_id)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for index, sent in enumerate(sents):
        tts.read_to_file(text_list=[sent], file_location=output_dir + "/{}.wav".format(index))


if __name__ == '__main__':
    save_weights("taco_libri")

    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.isdir("audios"):
        os.makedirs("audios")

    read_texts(model_id="fast_libri",
               sentence=["Hello world, I am a synthesis voice."],
               device=exec_device,
               speaker_embedding="default_speaker_embedding.pt",
               filename="audios/fast_libri.wav")

    read_texts(model_id="fast_nancy",
               sentence=["Hello world, I am a synthesis voice."],
               device=exec_device,
               speaker_embedding="default_speaker_embedding.pt",
               filename="audios/fast_nancy.wav")

    read_texts(model_id="taco_libri",
               sentence=["Hello world, I am a synthesis voice."],
               device=exec_device,
               speaker_embedding="default_speaker_embedding.pt",
               filename="audios/fast_libri.wav")

    read_texts(model_id="taco_nancy",
               sentence=["Hello world, I am a synthesis voice."],
               device=exec_device,
               speaker_embedding="default_speaker_embedding.pt",
               filename="audios/fast_nancy.wav")
