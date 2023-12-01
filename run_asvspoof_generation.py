import os

import librosa
import soundfile as sf
from tqdm import tqdm

from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface
from Utility.utils import float2pcm

PATH_TO_MLS_ENGLISH_TRAIN = "/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_english/train"
PATH_TO_GENERATION_FILE = "p1_ttsvc_surrogate.tsv"
PATH_TO_OUTPUT_DIR = "asv_spoof_outputs_no_pros"
DEVICE = "cuda"


def build_path_to_transcript_dict_mls_english():
    path_to_transcript = dict()
    with open(os.path.join(PATH_TO_MLS_ENGLISH_TRAIN, "transcripts.txt"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            fields = line.split("\t")
            wav_folders = fields[0].split("_")
            wav_path = f"{PATH_TO_MLS_ENGLISH_TRAIN}/audio/{wav_folders[0]}/{wav_folders[1]}/{fields[0]}.flac"
            path_to_transcript[wav_path] = fields[1]
    return path_to_transcript


if __name__ == '__main__':
    print("loading model...")
    tts = ToucanTTSInterface(device=DEVICE, tts_model_path="ASVSpoof")
    print("prepare path to transcript lookup...")
    path_to_transcript_dict = build_path_to_transcript_dict_mls_english()
    filename_to_path = dict()
    for p in path_to_transcript_dict:
        filename_to_path[p.split("/")[-1].rstrip(".flac")] = p
    with open(PATH_TO_GENERATION_FILE, "r") as file:
        generation_list = file.read().split("\n")
    os.makedirs(PATH_TO_OUTPUT_DIR, exist_ok=True)
    print("generating audios...")
    for generation_item in tqdm(generation_list):
        if generation_item == "":
            continue
        speaker_id, voice_sources, _, prosody_source, output_name = generation_item.split()
        voice_source_list = voice_sources.split(",")
        transcript = path_to_transcript_dict[filename_to_path[prosody_source]]
        source_list = list()
        for source in voice_source_list:
            source_list.append(filename_to_path[source])
        tts.set_utterance_embedding(path_to_reference_audio=source_list)
        cloned_utterance = tts(transcript)
        resampled_utt = librosa.resample(cloned_utterance, orig_sr=24000, target_sr=16000)
        sf.write(file=f"{PATH_TO_OUTPUT_DIR}/" + output_name + ".flac", data=float2pcm(resampled_utt), samplerate=16000, subtype="PCM_16")
