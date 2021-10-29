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

    read_texts(model_id="fast_karlsson",
               sentence=[
                   "Einst stritten sich Nordwind und Sonne, wer von ihnen beiden wohl der Stärkere wäre, als ein Wanderer, der in einen warmen Mantel gehüllt war, des Weges daherkam."],
               device=exec_device,
               filename="audios/1_1.wav")

    read_texts(model_id="fast_karlsson",
               sentence=["Sie wurden einig, dass derjenige für den Stärkeren gelten sollte, der den Wanderer zwingen würde, seinen Mantel abzunehmen."],
               device=exec_device,
               filename="audios/1_2.wav")

    read_texts(model_id="fast_karlsson",
               sentence=["Der Nordwind blies mit aller Macht, aber je mehr er blies, desto fester hüllte sich der Wanderer in seinen Mantel ein."],
               device=exec_device,
               filename="audios/1_3.wav")

    read_texts(model_id="fast_karlsson",
               sentence=["Endlich gab der Nordwind den Kampf auf."],
               device=exec_device,
               filename="audios/1_4.wav")

    read_texts(model_id="fast_karlsson",
               sentence=["Nun erwärmte die Sonne die Luft mit ihren freundlichen Strahlen, und schon nach wenigen Augenblicken zog der Wanderer seinen Mantel aus."],
               device=exec_device,
               filename="audios/1_5.wav")

    read_texts(model_id="fast_karlsson",
               sentence=["Da musste der Nordwind zugeben, dass die Sonne von ihnen beiden der Stärkere war."],
               device=exec_device,
               filename="audios/1_6.wav")

    read_texts(model_id="fast_low",
               sentence=[
                   "Einst stritten sich Nordwind und Sonne, wer von ihnen beiden wohl der Stärkere wäre, als ein Wanderer, der in einen warmen Mantel gehüllt war, des Weges daherkam."],
               device=exec_device,
               filename="audios/2_1.wav")

    read_texts(model_id="fast_low",
               sentence=["Sie wurden einig, dass derjenige für den Stärkeren gelten sollte, der den Wanderer zwingen würde, seinen Mantel abzunehmen."],
               device=exec_device,
               filename="audios/2_2.wav")

    read_texts(model_id="fast_low",
               sentence=["Der Nordwind blies mit aller Macht, aber je mehr er blies, desto fester hüllte sich der Wanderer in seinen Mantel ein."],
               device=exec_device,
               filename="audios/2_3.wav")

    read_texts(model_id="fast_low",
               sentence=["Endlich gab der Nordwind den Kampf auf."],
               device=exec_device,
               filename="audios/2_4.wav")

    read_texts(model_id="fast_low",
               sentence=["Nun erwärmte die Sonne die Luft mit ihren freundlichen Strahlen, und schon nach wenigen Augenblicken zog der Wanderer seinen Mantel aus."],
               device=exec_device,
               filename="audios/2_5.wav")

    read_texts(model_id="fast_low",
               sentence=["Da musste der Nordwind zugeben, dass die Sonne von ihnen beiden der Stärkere war."],
               device=exec_device,
               filename="audios/2_6.wav")
