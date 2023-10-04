import os
import sys
import warnings

import torch

from TTSInferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface
from Utility.storage_config import MODELS_DIR

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)

    PATH_TO_TTS_MODEL = os.path.join(MODELS_DIR, "ToucanTTS_Nancy", "best.pt")
    PATH_TO_REFERENCE_SPEAKER = ""  # audios/speaker_references_for_testing/female_high_voice.wav  audios/speaker_references_for_testing/male_low_voice.wav
    LANGUAGE = "en"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tts = ToucanTTSInterface(device=device, tts_model_path=PATH_TO_TTS_MODEL)
    tts.set_language(lang_id=LANGUAGE)
    if PATH_TO_REFERENCE_SPEAKER != "":
        if os.path.exists(PATH_TO_REFERENCE_SPEAKER):
            tts.set_utterance_embedding(PATH_TO_REFERENCE_SPEAKER)
        else:
            print(f"\n\nFile {PATH_TO_REFERENCE_SPEAKER} could not be found, please check for typos and re-run. Using default for now.\n\n")

    print("Loading the following configuration:")
    print(f"\tTTS Model:          {PATH_TO_TTS_MODEL}")
    print(f"\tReference Audio:    {PATH_TO_REFERENCE_SPEAKER}")
    print(f"\tLanguage Used:      {LANGUAGE}")
    print(f"\tDevice Used:        {device}")

    while True:
        text = input("\n\nWhat should I say? (or 'exit')\n")
        if text == "exit":
            sys.exit()
        tts.read_aloud(text,
                       view=True,
                       blocking=False,
                       duration_scaling_factor=1.0,
                       energy_variance_scale=1.0,
                       pitch_variance_scale=1.0)
