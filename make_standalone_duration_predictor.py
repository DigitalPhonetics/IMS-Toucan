import torch

from InferenceInterfaces.LJSpeech_FastSpeechInference import LJSpeech_FastSpeechInference
from StandaloneDurationPredictor.ProcessText import TextFrontend
from StandaloneDurationPredictor.StandaloneDurationPredictor import StandaloneDurationPredictor

if __name__ == '__main__':
    # extract the duration predictor from trained fastspeech model and save it
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = LJSpeech_FastSpeechInference(device=device, speaker_embedding=None)
    duration_predictor = StandaloneDurationPredictor()
    duration_predictor.encoder = tts.phone2mel.encoder
    duration_predictor.duration_predictor = tts.phone2mel.duration_predictor
    torch.save({"dp": duration_predictor.state_dict()}, "StandaloneDurationPredictor/duration_predictor_ljspeech.pt")

    # verify it works by loading it
    del tts
    del duration_predictor
    text = "Hello world, this is a sentence"
    dp = StandaloneDurationPredictor(path_to_model="StandaloneDurationPredictor/duration_predictor_ljspeech.pt")
    text2phone = TextFrontend(language="en", use_panphon_vectors=False, use_word_boundaries=False, use_explicit_eos=False)
    phones = text2phone.string_to_tensor(text).squeeze(0).long()
    print(text2phone.get_phone_string(text))
    print(dp(phones))
