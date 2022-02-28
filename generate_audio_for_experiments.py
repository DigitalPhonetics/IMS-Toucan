import os
import random

import torch
from tqdm import tqdm

from InferenceInterfaces.InferenceFastSpeech2 import InferenceFastSpeech2

torch.manual_seed(131714)
random.seed(131714)
torch.random.manual_seed(131714)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # hardcoded gpu ID, be careful with this script

###############################################################################################################################################

os.makedirs("experiment_audios/german/low_diff", exist_ok=True)

tts_low_diff = InferenceFastSpeech2(device="cuda" if torch.cuda.is_available() else "cpu", model_name="Meta_joint_finetune_german", language="de")
tts_low_diff.set_utterance_embedding("audios/german_female.wav")

with open("experiment_audios/german/low_diff/transcripts_in_kaldi_format.txt", encoding="utf8", mode="r") as f:
    trans = f.read()
for index, line in enumerate(tqdm(trans.split("\n"))):
    if line.strip() != "":
        assert line.startswith(f"{index} ")
        text = line.lstrip(f"{index} ")
        tts_low_diff.read_to_file([text], silent=True, file_location=f"experiment_audios/german/low_diff/{index}.wav")
