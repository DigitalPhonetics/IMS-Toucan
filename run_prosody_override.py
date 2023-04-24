import torch

from InferenceInterfaces.UtteranceCloner import UtteranceCloner

if __name__ == '__main__':
    uc = UtteranceCloner(model_id="Meta", device="cuda" if torch.cuda.is_available() else "cpu")

    # What is said in path_to_reference_audio_for_intonation has to match the text in the reference_transcription exactly!
    uc.clone_utterance(path_to_reference_audio_for_intonation="audios/test.wav",
                       path_to_reference_audio_for_voice="audios/test.wav",  # the two reference audios can be the same, but don't have to be
                       transcription_of_intonation_reference="Hello world, this is a test.",
                       filename_of_result="audios/test_cloned.wav",
                       lang="en")

    # Have multiple voices speak with the exact same intonation simultaneously
    uc.biblical_accurate_angel_mode(path_to_reference_audio_for_intonation="audios/test.wav",
                                    transcription_of_intonation_reference="Hello world, this is a test.",
                                    list_of_speaker_references_for_ensemble=["audios/speaker_references_for_testing/female_high_voice.wav",
                                                                             "audios/speaker_references_for_testing/female_mid_voice.wav",
                                                                             "audios/speaker_references_for_testing/male_low_voice.wav"],
                                    filename_of_result="audios/test_cloned_angelic.wav",
                                    lang="en")
