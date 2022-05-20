import os
import shutil

import soundfile

from InferenceInterfaces.InferenceFastSpeech2 import InferenceFastSpeech2
from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend

# =================================================
# remember to call run_weight_averaging.py before to prepare the inference checkpoint file
model_id = "Online"
name_of_output_dir = "audios/test_Online"
# =================================================


tf = ArticulatoryCombinedTextFrontend(language="en")
tts = InferenceFastSpeech2(device="cpu", model_name=model_id)
tts.set_language("en")
os.makedirs(name_of_output_dir, exist_ok=True)

for speaker in os.listdir("audios/speaker_references_for_testing"):
    shutil.copy(f"audios/speaker_references_for_testing/{speaker}", name_of_output_dir + "/" + speaker.rstrip('.wav') + "_original.wav")
    tts.set_utterance_embedding(f"audios/speaker_references_for_testing/{speaker}")
    phones = tf.get_phone_string("Hello there, this is a sentence that should be sufficiently long to see whether the speaker is captured adequately.",
                                 for_plot_labels=False)
    wave = tts(phones,
               view=False,
               input_is_phones=True)
    soundfile.write(file=f"{name_of_output_dir}/{speaker.rstrip('.wav')}_synthesized.wav", data=wave.cpu().numpy(), samplerate=48000)
