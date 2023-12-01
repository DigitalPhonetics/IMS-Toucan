import os

import librosa
import soundfile as sf
import torch
from tqdm import tqdm

from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface
from Utility.utils import float2pcm

PATH_TO_GENERATION_FILE = "p1_ttsvc_surrogate.tsv"
PATH_TO_OUTPUT_DIR = "asv_spoof_outputs_no_pros"
DEVICE = "cuda"

if __name__ == '__main__':
    tts = ToucanTTSInterface(device=DEVICE, tts_model_path="ASVSpoof")
    path_to_transcript_dict = torch.load("mls_transcript_cache.pt")
    filename_to_path = dict()
    for p in path_to_transcript_dict:
        filename_to_path[p.split("/")[-1].rstrip(".flac")] = p
    with open(PATH_TO_GENERATION_FILE, "r") as file:
        generation_list = file.read().split("\n")
    os.makedirs(PATH_TO_OUTPUT_DIR, exist_ok=True)

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
        sf.write(file=f"{PATH_TO_OUTPUT_DIR}/" + output_name + ".flac",
                 data=float2pcm(resampled_utt),
                 samplerate=16000,
                 subtype="PCM_16")
