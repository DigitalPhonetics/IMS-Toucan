import os
import random

import torch

from InferenceInterfaces.InferenceFastSpeech2 import InferenceFastSpeech2

torch.manual_seed(131714)
random.seed(131714)
torch.random.manual_seed(131714)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # hardcoded gpu ID, be careful with this script

speaker_references = ["spanish.wav",
                      "russian.wav",
                      "portuguese.flac",
                      "polish.flac",
                      "italian.flac",
                      "hungarian.wav",
                      "greek.wav",
                      "german.wav",
                      "french.wav",
                      "finnish.wav",
                      "english.wav",
                      "dutch.wav"]

lang_sents = {
    "de": "Klebe das Blatt auf den dunkelblauen Hintergrund. Das Kanu aus Birkenholz glitt über die glatten Planken.",
    "es": "Pega la hoja al fondo azul oscuro. La canoa de abedul se deslizó sobre los lisos tablones.",
    "ru": "Приклейте лист к темно-синему фону. Березовое каноэ скользило по гладким доскам.",
    "pt": "Colar a folha ao fundo azul escuro. A canoa de bétula deslizou sobre as tábuas lisas.",
    "pl": "Przyklej arkusz do ciemnoniebieskiego tła. Brzozowy kajak ślizgał się po gładkich deskach.",
    "it": "Incollare il foglio allo sfondo blu scuro. La canoa di betulla scivolava sulle tavole lisce.",
    "hu": "Ragassza a lapot a sötétkék háttérre. A nyírfa kenu csúszott a sima deszkákon.",
    "el": "Κολλήστε το φύλλο στο σκούρο μπλε φόντο. Το κανό από σημύδα γλίστρησε στις λείες σανίδες.",
    "fr": "Collez la feuille sur le fond bleu foncé. Le canoë en bouleau a glissé sur les planches lisses.",
    "fi": "Liimaa arkki tummansiniseen taustaan. Koivukanootti liukui sileillä lankuilla.",
    "en": "Glue the sheet to the dark blue background. The birch canoe slid on the smooth planks.",
    "nl": "Lijm het blad op de donkerblauwe achtergrond. De berkenhouten kano gleed over de gladde planken."
    }

os.makedirs("experiment_audios/speakers_for_plotting", exist_ok=True)

tts = InferenceFastSpeech2(device="cuda" if torch.cuda.is_available() else "cpu", model_name="Meta")

for speaker_ref in speaker_references:
    tts.set_utterance_embedding(speaker_ref)
    for lang in lang_sents:
        tts.set_language(lang)
        tts.read_to_file(lang_sents[lang], f"experiment_audios/speakers_for_plotting/{speaker_ref.split('.')[0]}_{lang}.wav")
