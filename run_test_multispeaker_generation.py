import os
import shutil

from InferenceInterfaces.InferenceFastSpeech2 import InferenceFastSpeech2

# =================================================
# remember to call run_weight_averaging.py before to prepare the inference checkpoint file
model_id = "Online"
name_of_output_dir = "audios/test_Online"
# =================================================

tts = InferenceFastSpeech2(device="cpu", model_name=model_id)
tts.set_language("en")
os.makedirs(name_of_output_dir, exist_ok=True)

for speaker in os.listdir("audios/speaker_references_for_testing"):
    shutil.copy(f"audios/speaker_references_for_testing/{speaker}", name_of_output_dir + "/" + speaker.rstrip('.wav') + "_original.wav")
    tts.set_utterance_embedding(f"audios/speaker_references_for_testing/{speaker}")
    tts.read_to_file(text_list=["Hello there, this is a sentence that should be sufficiently long to see whether the speaker is captured adequately."],
                     file_location=f"{name_of_output_dir}/{speaker.rstrip('.wav')}_synthesized.wav")
