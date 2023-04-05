import librosa
import soundfile as sf
import torch
from tqdm import tqdm

from Preprocessing.TextFrontend import get_feature_to_index_lookup
from Utility.path_to_transcript_dicts import *


def make_sielce_cleaned_versions(train_sets):
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True  # torch 1.9 has a bug in the hub loading, this is a workaround
    # careful: assumes 16kHz or 8kHz audio
    silero_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                         model='silero_vad',
                                         force_reload=False,
                                         onnx=False,
                                         verbose=False)
    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils
    torch.set_grad_enabled(True)  # finding this issue was very infuriating: silero sets
    # this to false globally during model loading rather than using inference mode or no_grad
    device = "cuda" if torch.cuda.is_available() else "cpu"
    silero_model = silero_model.to(device)

    for train_set in train_sets:
        for index in tqdm(range(len(train_set))):
            filepath = train_set.datapoints[index][8]
            phonemes = train_set.datapoints[index][0]
            speech_length = train_set.datapoints[index][3]
            durations = train_set.datapoints[index][4]
            cumsum = 0
            legal_silences = list()
            for phoneme_index, phone in enumerate(phonemes):
                if phone[get_feature_to_index_lookup()["silence"]] == 1 or phone[get_feature_to_index_lookup()["end of sentence"]] == 1 or phone[get_feature_to_index_lookup()["questionmark"]] == 1 or phone[get_feature_to_index_lookup()["exclamationmark"]] == 1 or phone[get_feature_to_index_lookup()["fullstop"]] == 1:
                    legal_silences.append([cumsum, cumsum + durations[phoneme_index]])
                cumsum = cumsum + durations[phoneme_index]
            wave, sr = sf.read(filepath)
            resampled_wave = librosa.resample(wave, orig_sr=sr, target_sr=16000)
            with torch.inference_mode():
                speech_timestamps = get_speech_timestamps(torch.Tensor(resampled_wave).to(device), silero_model, sampling_rate=16000)
            silences = list()
            prev_end = 0
            for speech_segment in speech_timestamps:
                if prev_end != 0:
                    silences.append([prev_end, speech_segment["start"]])
                prev_end = speech_segment["end"]
            # at this point we know all the silences and we know the legal silences.
            # We have to transform them both into ratios, so we can compare them.
            # If a silence overlaps with a legal silence, it can stay.
            illegal_silences = list()
            for silence in silences:
                illegal = True
                start = silence[0] / len(resampled_wave)
                end = silence[1] / len(resampled_wave)
                for legal_silence in legal_silences:
                    legal_start = legal_silence[0] / speech_length
                    legal_end = legal_silence[1] / speech_length
                    if legal_start < start < legal_end or legal_start < end < legal_end:
                        illegal = False
                        break
                if illegal:
                    # If it is an illegal silence, it is marked for removal in the original wave according to ration with real samplingrate.
                    illegal_silences.append([start, end])

            # print(f"{len(illegal_silences)} illegal silences detected. ({len(silences) - len(illegal_silences)} legal silences left)")
            wave = list(wave)
            orig_wave_length = len(wave)
            for illegal_silence in reversed(illegal_silences):
                wave = wave[:int(illegal_silence[0] * orig_wave_length)] + wave[int(illegal_silence[1] * orig_wave_length):]
            # Audio with illegal silences removed will be saved into a new directory.
            new_filepath_list = filepath.split("/")
            new_filepath_list[-2] = new_filepath_list[-2] + "_silence_removed"
            os.makedirs("/".join(new_filepath_list[:-1]), exist_ok=True)
            sf.write("/".join(new_filepath_list), wave, sr)
