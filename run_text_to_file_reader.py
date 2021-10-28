import os

import torch

from InferenceInterfaces.Eva_FastSpeech2 import Eva_FastSpeech2
from InferenceInterfaces.Eva_Tacotron2 import Eva_Tacotron2
from InferenceInterfaces.HokusPokus_FastSpeech2 import HokusPokus_FastSpeech2
from InferenceInterfaces.HokusPokus_Tacotron2 import HokusPokus_Tacotron2
from InferenceInterfaces.Karlsson_FastSpeech2 import Karlsson_FastSpeech2
from InferenceInterfaces.Karlsson_Tacotron2 import Karlsson_Tacotron2
from InferenceInterfaces.LowRes_FastSpeech2 import LowRes_FastSpeech2 as fast_low
from InferenceInterfaces.LowRes_Tacotron2 import LowRes_Tacotron2 as taco_low
from InferenceInterfaces.Nancy_FastSpeech2 import Nancy_FastSpeech2
from InferenceInterfaces.Nancy_Tacotron2 import Nancy_Tacotron2

tts_dict = {
    "fast_nancy": Nancy_FastSpeech2,
    "fast_hokus": HokusPokus_FastSpeech2,

    "taco_nancy": Nancy_Tacotron2,
    "taco_hokus": HokusPokus_Tacotron2,

    "taco_low": taco_low,
    "fast_low": fast_low,

    "taco_eva": Eva_Tacotron2,
    "fast_eva": Eva_FastSpeech2,

    "taco_karlsson": Karlsson_Tacotron2,
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

    read_texts(model_id="fast_nancy",
               sentence=["Hello world, I am a synthesis voice."],
               device=exec_device,
               filename="audios/fast_nancy.wav")

    read_texts(model_id="taco_nancy",
               sentence=["Hello world, I am a synthesis voice."],
               device=exec_device,
               filename="audios/fast_nancy.wav")
