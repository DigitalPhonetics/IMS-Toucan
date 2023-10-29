import torch

from InferenceInterfaces.UtteranceCloner import UtteranceCloner

if __name__ == '__main__':
    uc = UtteranceCloner(model_id="Nancy", device="cuda" if torch.cuda.is_available() else "cpu")

    # What is said in path_to_reference_audio_for_intonation has to match the text in the reference_transcription exactly!
    uc.clone_utterance(path_to_reference_audio_for_intonation="audios/speaker_references_for_testing/sad.wav",
                       path_to_reference_audio_for_voice="audios/speaker_references_for_testing/female_mid_voice.wav",  # the two reference audios can be the same, but don't have to be
                       transcription_of_intonation_reference="This report is due tomorrow.",
                       filename_of_result="audios/test_cloned.wav",
                       lang="en")
