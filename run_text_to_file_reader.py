import os

import torch

from InferenceInterfaces.Meta_FastSpeech2 import Meta_FastSpeech2
from InferenceInterfaces.MultiEnglish_FastSpeech2 import MultiEnglish_FastSpeech2
from InferenceInterfaces.MultiGerman_FastSpeech2 import MultiGerman_FastSpeech2

tts_dict = {
    "fast_meta"   : Meta_FastSpeech2,
    "fast_german" : MultiGerman_FastSpeech2,
    "fast_english": MultiEnglish_FastSpeech2
    }


def read_texts(model_id, sentence, filename, device="cpu"):
    tts = tts_dict[model_id](device=device)
    if type(sentence) == str:
        sentence = [sentence]
    tts.read_to_file(text_list=sentence, file_location=filename)
    del tts


def read_texts_as_ensemble(model_id, sentence, filename, device="cpu"):
    """
    for this function, the filename should NOT contain the .wav ending, it's added automatically
    """
    tts = tts_dict[model_id](device=device)
    if type(sentence) == str:
        sentence = [sentence]
    for index in range(10):
        tts.default_utterance_embedding = torch.zeros(704).float().random_(-40, 40).to(device)
        tts.read_to_file(text_list=sentence, file_location=filename + f"_{index}" + ".wav")


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


def read_contrastive_focus_sentences(model_id, device):
    tts = tts_dict[model_id](device=device)

    with open("Utility/contrastive_focus_test_sentences.txt", "r", encoding="utf8") as f:
        sents = f.read().split("\n")
    output_dir = "audios/focus_{}".format(model_id)
    os.makedirs(output_dir, exist_ok=True)
    for index, sent in enumerate(sents):
        tts.read_to_file(text_list=[sent], file_location=output_dir + "/{}.wav".format(index))


if __name__ == '__main__':
    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("audios", exist_ok=True)

    read_texts_as_ensemble(model_id="fast_meta",
                           sentence=["Hello world, this is a test."],
                           device=exec_device,
                           filename="audios/ensemble")
