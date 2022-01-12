import os

import torch

from InferenceInterfaces.Karlsson_FastSpeech2 import Karlsson_FastSpeech2
from InferenceInterfaces.Nancy_FastSpeech2 import Nancy_FastSpeech2

tts_dict = {
    "fast_nancy"   : Nancy_FastSpeech2,
    "fast_karlsson": Karlsson_FastSpeech2,
    }


def read_texts(model_id, sentence, filename, device="cpu"):
    tts = tts_dict[model_id](device=device)
    if type(sentence) == str:
        sentence = [sentence]
    tts.read_to_file(text_list=sentence, file_location=filename)
    del tts


def save_weights(model_id):
    tts_dict[model_id](device="cpu").save_pretrained_weights()


def read_harvard_sentences(model_id, device):
    tts = tts_dict[model_id](device=device)

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

    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.isdir("audios"):
        os.makedirs("audios")

    read_texts(model_id="fast_karlsson",
               sentence=[
                   "Einst stritten sich Nordwind und Sonne, wer von ihnen beiden wohl der Stärkere wäre, als ein Wanderer, der in einen warmen Mantel gehüllt war, des Weges daherkam."],
               device=exec_device,
               filename="audios/1.wav")

    read_texts(model_id="fast_nancy",
               sentence=["This is an English sentence, I really hope that it works."],
               device=exec_device,
               filename="audios/2.wav")
