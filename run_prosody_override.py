import torch

from InferenceInterfaces.UtteranceCloner import UtteranceCloner

if __name__ == '__main__':
    gpu_id = 7
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    uc = UtteranceCloner(model_id="CFM/epd_c8", device="cuda" if torch.cuda.is_available() else "cpu")

    sentences1 = ["Galleries are free on thursdays",
                              "Jessie dunked the basketball in the hoop"]
    sentences2= ["Builders put scaffolding around the windows",
                           "Michael drinks his tea with milk"]

    for k, sentence1 in enumerate(sentences1):
        for i in range(3):
            path = f"audios/Study/Human2/HUMAN-speaker1-sentence{k}_{i + 1}.wav"
            # What is said in path_to_reference_audio_for_intonation has to match the text in the reference_transcription exactly!
            uc.clone_utterance(path_to_reference_audio_for_intonation=path,
                            path_to_reference_audio_for_voice="audios/Study/Human/male.wav",  # the two reference audios can be the same, but don't have to be
                            transcription_of_intonation_reference=sentence1 + ".",
                            filename_of_result=f"audios/cloned3/HUMAN-speaker1-sentence{k}_{i + 1}.wav",
                            lang="eng")
    for k, sentence2 in enumerate(sentences2):
       
        ref_sentence = sentence2 + "."
        for i in range(3):
            path = f"audios/Study/Human2/HUMAN-speaker2-sentence{k}_{i + 1}.wav"
            # What is said in path_to_reference_audio_for_intonation has to match the text in the reference_transcription exactly!
            uc.clone_utterance(path_to_reference_audio_for_intonation=path,
                            path_to_reference_audio_for_voice="audios/Study/Human/female.wav",  # the two reference audios can be the same, but don't have to be
                            transcription_of_intonation_reference=ref_sentence,
                            filename_of_result=f"audios/cloned3/HUMAN-speaker2-sentence{k}_{i + 1}.wav",
                            lang="eng")
