import os
import random
import shutil

import torch
from tqdm import tqdm

from InferenceInterfaces.InferenceFastSpeech2 import InferenceFastSpeech2
from run_utterance_cloner import UtteranceCloner

torch.manual_seed(131714)
random.seed(131714)
torch.random.manual_seed(131714)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # hardcoded gpu ID, be careful with this script

porto_speaker_reference = "audios/portuguese.flac"

###############################################################################################################################################


# multiling

os.makedirs("experiment_audios/russian/low_diff", exist_ok=True)

tts_low_diff = InferenceFastSpeech2(device="cuda" if torch.cuda.is_available() else "cpu", model_name="Meta_joint_finetune_russian", language="ru")
tts_low_diff.set_utterance_embedding(porto_speaker_reference)

with open("experiment_audios/russian/low_diff/transcripts_in_kaldi_format.txt", encoding="utf8", mode="r") as f:
    trans = f.read()
for index, line in enumerate(tqdm(trans.split("\n"))):
    if line.strip() != "":
        assert line.startswith(f"{index} ")
        text = line.lstrip(f"{index} ")
        tts_low_diff.read_to_file([text], silent=True, file_location=f"experiment_audios/russian/low_diff/{index}.wav")

###############################################################################################################################################

os.makedirs("experiment_audios/german/low_diff", exist_ok=True)

tts_low_diff = InferenceFastSpeech2(device="cuda" if torch.cuda.is_available() else "cpu", model_name="Meta_joint_finetune_german", language="de")
tts_low_diff.set_utterance_embedding(porto_speaker_reference)

with open("experiment_audios/german/low_diff/transcripts_in_kaldi_format.txt", encoding="utf8", mode="r") as f:
    trans = f.read()
for index, line in enumerate(tqdm(trans.split("\n"))):
    if line.strip() != "":
        assert line.startswith(f"{index} ")
        text = line.lstrip(f"{index} ")
        tts_low_diff.read_to_file([text], silent=True, file_location=f"experiment_audios/german/low_diff/{index}.wav")

###############################################################################################################################################

# cloning

os.makedirs("experiment_audios/adept/human", exist_ok=True)
os.makedirs("experiment_audios/adept/diff_voice_same_style", exist_ok=True)
os.makedirs("experiment_audios/adept/same_voice_same_style", exist_ok=True)
os.makedirs("experiment_audios/adept/same_voice_diff_style", exist_ok=True)

uc = UtteranceCloner(model_id="Meta", device="cuda" if torch.cuda.is_available() else "cpu")
tts = InferenceFastSpeech2(device="cuda" if torch.cuda.is_available() else "cpu", model_name="Meta", language="en")

root = "/mount/resources/speech/corpora/ADEPT/"
transcript_paths = list()
audio_paths = list()

transcript_paths_study = list()
audio_paths_study = list()

for prop in os.listdir(root + "txt"):
    for cat in os.listdir(root + f"txt/{prop}"):
        for trans in os.listdir(root + f"txt/{prop}/{cat}"):
            transcript_paths.append(root + f"txt/{prop}/{cat}/{trans}")
            audio_paths.append(root + f"wav_44khz/{prop}/{cat}/{trans.split('.')[0]}.wav")
            if prop == "marked_tonicity" and cat == "interpretation_2":
                transcript_paths_study.append(root + f"txt/{prop}/{cat}/{trans}")
                audio_paths_study.append(root + f"wav_44khz/{prop}/{cat}/{trans.split('.')[0]}.wav")

kaldi_style_transcript = ""
for index in tqdm(list(range(len(transcript_paths)))):
    with open(transcript_paths[index], encoding="utf8", mode="r") as trans_file:
        text = trans_file.read().strip()
    shutil.copy(audio_paths[index], f"experiment_audios/adept/human/{index}.wav")
    kaldi_style_transcript += f"{index} {text}\n"
    audio_path = audio_paths[index]
    uc.clone_utterance(path_to_reference_audio=audio_path,
                       reference_transcription=text,
                       filename_of_result=f"experiment_audios/adept/diff_voice_same_style/{index}.wav",
                       clone_speaker_identity=False,
                       lang="en")
    uc.clone_utterance(path_to_reference_audio=audio_path,
                       reference_transcription=text,
                       filename_of_result=f"experiment_audios/adept/same_voice_same_style/{index}.wav",
                       clone_speaker_identity=True,
                       lang="en")
    tts.set_utterance_embedding(audio_path)
    tts.read_to_file([text], silent=True, file_location=f"experiment_audios/adept/same_voice_diff_style/{index}.wav")

with open("experiment_audios/adept/human/transcripts_in_kaldi_format.txt", encoding="utf8", mode="w") as f:
    f.write(kaldi_style_transcript)
with open("experiment_audios/adept/same_voice_diff_style/transcripts_in_kaldi_format.txt", encoding="utf8", mode="w") as f:
    f.write(kaldi_style_transcript)
with open("experiment_audios/adept/diff_voice_same_style/transcripts_in_kaldi_format.txt", encoding="utf8", mode="w") as f:
    f.write(kaldi_style_transcript)
with open("experiment_audios/adept/same_voice_same_style/transcripts_in_kaldi_format.txt", encoding="utf8", mode="w") as f:
    f.write(kaldi_style_transcript)

###############################################################################################################################################

os.makedirs("experiment_audios/adept_study/human", exist_ok=True)
os.makedirs("experiment_audios/adept_study/diff_voice_same_style", exist_ok=True)
os.makedirs("experiment_audios/adept_study/same_voice_same_style", exist_ok=True)
os.makedirs("experiment_audios/adept_study/same_voice_diff_style", exist_ok=True)

for index in tqdm(list(range(len(transcript_paths_study)))):
    with open(transcript_paths_study[index], encoding="utf8", mode="r") as trans_file:
        text = trans_file.read().strip()
    shutil.copy(audio_paths_study[index], f"experiment_audios/adept_study/human/{index}.wav")
    audio_path = audio_paths_study[index]
    uc.clone_utterance(path_to_reference_audio=audio_path,
                       reference_transcription=text,
                       filename_of_result=f"experiment_audios/adept_study/diff_voice_same_style/{index}.wav",
                       clone_speaker_identity=False,
                       lang="en")
    uc.clone_utterance(path_to_reference_audio=audio_path,
                       reference_transcription=text,
                       filename_of_result=f"experiment_audios/adept_study/same_voice_same_style/{index}.wav",
                       clone_speaker_identity=True,
                       lang="en")
    tts.set_utterance_embedding(audio_path)
    tts.read_to_file([text], silent=True, file_location=f"experiment_audios/adept_study/same_voice_diff_style/{index}.wav")

###############################################################################################################################################

# the rest is populated by hand or low priority

os.makedirs("experiment_audios/russian_study/human", exist_ok=True)
os.makedirs("experiment_audios/russian_study/low", exist_ok=True)
os.makedirs("experiment_audios/russian_study/single", exist_ok=True)

os.makedirs("experiment_audios/german_study/human", exist_ok=True)
os.makedirs("experiment_audios/german_study/low", exist_ok=True)
os.makedirs("experiment_audios/german_study/single", exist_ok=True)

os.makedirs("experiment_audios/speakers_for_plotting", exist_ok=True)
