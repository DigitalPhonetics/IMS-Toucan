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


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.isdir("audios"):
        os.makedirs("audios")

    read_texts(model_id="trans_thorsten",
               sentence=["Im Frühling denkt das Rößlein, wer nicht leiden will, muss schön sein!"],
               filename="audios/thorsten.wav",
               device=device)
