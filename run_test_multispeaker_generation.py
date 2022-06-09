if __name__ == '__main__':
    from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
    from InferenceInterfaces.InferenceFastSpeech2 import InferenceFastSpeech2
    import os
    import shutil

    # for some reason, the espeak installation is not found if the order of the first two imports is switched, so
    # they are part of the code now that autoformat doesn't sort them alphabetically.

    # =================================================
    # remember to call run_weight_averaging.py before to prepare the inference checkpoint file
    model_id = "LibriGST"
    name_of_output_dir = "audios/test_gst"
    # =================================================

    tts = InferenceFastSpeech2(device="cpu", model_name=model_id)
    tts.set_language("en")

    os.makedirs(name_of_output_dir, exist_ok=True)
    ArticulatoryCombinedTextFrontend(language="en")  # this is 100% unnecessary, but if it is not there,
    # a random error will occur, because something weird is happening in the import of espeak, and it doesn't
    # like if its import is embedded too deeply.

    for speaker in os.listdir("audios/speaker_references_for_testing"):
        shutil.copy(f"audios/speaker_references_for_testing/{speaker}", name_of_output_dir + "/" + speaker.rstrip('.wav') + "_original.wav")
        tts.set_utterance_embedding(f"audios/speaker_references_for_testing/{speaker}")
        tts.read_to_file(text_list=["Hello there, this is a sentence that should be sufficiently long to see whether the speaker is captured adequately."],
                         file_location=f"{name_of_output_dir}/{speaker.rstrip('.wav')}_synthesized.wav")
