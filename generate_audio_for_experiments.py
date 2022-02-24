import os
import random
import shutil

import soundfile
import torch
from tqdm import tqdm

from InferenceInterfaces.InferenceFastSpeech2 import InferenceFastSpeech2
from Utility.path_to_transcript_dicts import build_path_to_transcript_dict_css10ru
from Utility.path_to_transcript_dicts import build_path_to_transcript_dict_karlsson
from Utility.path_to_transcript_dicts import build_path_to_transcript_dict_mls_portuguese
from run_utterance_cloner import UtteranceCloner

torch.manual_seed(131714)
random.seed(131714)
torch.random.manual_seed(131714)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "8"  # hardcoded gpu ID, be careful with this script

russian_ptt = build_path_to_transcript_dict_css10ru()
german_ptt = build_path_to_transcript_dict_karlsson()
porto_ptt = build_path_to_transcript_dict_mls_portuguese()

longest_sample_in_porto = [None, 0]
for path in tqdm(porto_ptt):
    wave, sr = soundfile.read(path)
    if len(wave) > longest_sample_in_porto[1]:
        longest_sample_in_porto[0] = path
        longest_sample_in_porto[1] = len(wave)
        # so we get a good speaker reference
porto_speaker_reference = longest_sample_in_porto[0]

###############################################################################################################################################


# multiling


os.makedirs("experiment_audios/russian/human", exist_ok=True)
os.makedirs("experiment_audios/russian/low_same", exist_ok=True)
os.makedirs("experiment_audios/russian/low_diff", exist_ok=True)
os.makedirs("experiment_audios/russian/single", exist_ok=True)

tts_single = InferenceFastSpeech2(device="cuda" if torch.cuda.is_available() else "cpu", model_name="RussianSingle", language="ru")
tts_low_same = InferenceFastSpeech2(device="cuda" if torch.cuda.is_available() else "cpu", model_name="Russian_low_resource", language="ru")
tts_low_diff = InferenceFastSpeech2(device="cuda" if torch.cuda.is_available() else "cpu", model_name="Russian_low_resource", language="ru")
tts_low_diff.set_utterance_embedding(porto_speaker_reference)

kaldi_style_transcript = ""
for index, path in enumerate(random.sample(list(russian_ptt.keys()), 3000)):
    # 3000 audios should be totally sufficient, and we won't have to wait for days
    shutil.copy(path, f"experiment_audios/russian/human/{index}.wav")
    kaldi_style_transcript += f"{index} {russian_ptt[path]}\n"
    tts_single.read_to_file([russian_ptt[path]], silent=True, file_location=f"experiment_audios/russian/single/{index}.wav")
    tts_low_same.read_to_file([russian_ptt[path]], silent=True, file_location=f"experiment_audios/russian/low_same/{index}.wav")
    tts_low_diff.read_to_file([russian_ptt[path]], silent=True, file_location=f"experiment_audios/russian/low_diff/{index}.wav")
with open("experiment_audios/russian/human/transcripts_in_kaldi_format.txt", encoding="utf8", mode="w") as f:
    f.write(kaldi_style_transcript)
with open("experiment_audios/russian/low_same/transcripts_in_kaldi_format.txt", encoding="utf8", mode="w") as f:
    f.write(kaldi_style_transcript)
with open("experiment_audios/russian/low_diff/transcripts_in_kaldi_format.txt", encoding="utf8", mode="w") as f:
    f.write(kaldi_style_transcript)
with open("experiment_audios/russian/single/transcripts_in_kaldi_format.txt", encoding="utf8", mode="w") as f:
    f.write(kaldi_style_transcript)

###############################################################################################################################################

os.makedirs("experiment_audios/german/human", exist_ok=True)
os.makedirs("experiment_audios/german/low_same", exist_ok=True)
os.makedirs("experiment_audios/german/low_diff", exist_ok=True)
os.makedirs("experiment_audios/german/single", exist_ok=True)

tts_single = InferenceFastSpeech2(device="cuda" if torch.cuda.is_available() else "cpu", model_name="GermanSingle", language="de")
tts_low_same = InferenceFastSpeech2(device="cuda" if torch.cuda.is_available() else "cpu", model_name="German_low_resource", language="de")
tts_low_diff = InferenceFastSpeech2(device="cuda" if torch.cuda.is_available() else "cpu", model_name="German_low_resource", language="de")
tts_low_diff.set_utterance_embedding(porto_speaker_reference)

kaldi_style_transcript = ""
for index, path in enumerate(random.sample(list(german_ptt.keys()), 3000)):
    # 3000 audios should be totally sufficient, and we won't have to wait for days
    shutil.copy(path, f"experiment_audios/german/human/{index}.wav")
    kaldi_style_transcript += f"{index} {german_ptt[path]}\n"
    tts_single.read_to_file([german_ptt[path]], silent=True, file_location=f"experiment_audios/german/single/{index}.wav")
    tts_low_same.read_to_file([german_ptt[path]], silent=True, file_location=f"experiment_audios/german/low_same/{index}.wav")
    tts_low_diff.read_to_file([german_ptt[path]], silent=True, file_location=f"experiment_audios/german/low_diff/{index}.wav")
with open("experiment_audios/german/human/transcripts_in_kaldi_format.txt", encoding="utf8", mode="w") as f:
    f.write(kaldi_style_transcript)
with open("experiment_audios/german/low_same/transcripts_in_kaldi_format.txt", encoding="utf8", mode="w") as f:
    f.write(kaldi_style_transcript)
with open("experiment_audios/german/low_diff/transcripts_in_kaldi_format.txt", encoding="utf8", mode="w") as f:
    f.write(kaldi_style_transcript)
with open("experiment_audios/german/single/transcripts_in_kaldi_format.txt", encoding="utf8", mode="w") as f:
    f.write(kaldi_style_transcript)

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
for index in range(len(transcript_paths)):
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

for index in range(len(transcript_paths_study)):
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

os.makedirs("experiment_audios/russian/human_study", exist_ok=True)
os.makedirs("experiment_audios/russian/low_study", exist_ok=True)
os.makedirs("experiment_audios/russian/single_study", exist_ok=True)

os.makedirs("experiment_audios/german/human_study", exist_ok=True)
os.makedirs("experiment_audios/german/low_study", exist_ok=True)
os.makedirs("experiment_audios/german/single_study", exist_ok=True)

os.makedirs("experiment_audios/speakers_for_plotting", exist_ok=True)
