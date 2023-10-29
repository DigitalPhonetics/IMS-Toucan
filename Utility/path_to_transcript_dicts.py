import glob
import os
import random
from pathlib import Path

import torch


# HELPER FUNCTIONS

def split_dictionary_into_chunks(input_dict, split_n):
    res = []
    new_dict = {}
    elements_per_dict = (len(input_dict.keys()) // split_n) + 1
    for k, v in input_dict.items():
        if len(new_dict) < elements_per_dict:
            new_dict[k] = v
        else:
            res.append(new_dict)
            new_dict = {k: v}
    res.append(new_dict)
    return res


def limit_to_n(path_to_transcript_dict, n=40000):
    # deprecated, we now just use the whole thing always, because there's a critical mass of data
    limited_dict = dict()
    if len(path_to_transcript_dict.keys()) > n:
        for key in random.sample(list(path_to_transcript_dict.keys()), n):
            limited_dict[key] = path_to_transcript_dict[key]
        return limited_dict
    else:
        return path_to_transcript_dict


def build_path_to_transcript_dict_multi_ling_librispeech_template(root):
    """
    https://arxiv.org/abs/2012.03411
    """
    path_to_transcript = dict()
    with open(os.path.join(root, "transcripts.txt"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            fields = line.split("\t")
            wav_folders = fields[0].split("_")
            wav_path = f"{root}/audio/{wav_folders[0]}/{wav_folders[1]}/{fields[0]}.flac"
            path_to_transcript[wav_path] = fields[1]
    return path_to_transcript


def build_path_to_transcript_dict_hui_template(root):
    """
    https://arxiv.org/abs/2106.06309
    """
    path_to_transcript = dict()
    for el in os.listdir(root):
        if os.path.isdir(os.path.join(root, el)):
            with open(os.path.join(root, el, "metadata.csv"), "r", encoding="utf8") as file:
                lookup = file.read()
            for line in lookup.split("\n"):
                if line.strip() != "":
                    norm_transcript = line.split("|")[1]
                    wav_path = os.path.join(root, el, "wavs", line.split("|")[0] + ".wav")
                    if os.path.exists(wav_path):
                        path_to_transcript[wav_path] = norm_transcript
    return path_to_transcript


# ENGLISH

def build_path_to_transcript_dict_mls_english(re_cache=False):
    lang = "english"
    root = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train"
    cache_path = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train/pttd_cache.pt"
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = build_path_to_transcript_dict_multi_ling_librispeech_template(root=root)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_gigaspeech(re_cache=False):
    root = "/mount/resources/speech/corpora/GigaSpeech/"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(os.path.join(root, "transcripts_only_clean_samples.txt"), "r", encoding="utf8") as file:
            lookup = file.read()
        for line in lookup.split("\n"):
            if line.strip() != "":
                fields = line.split("\t")
                norm_transcript = fields[1]
                wav_path = fields[0]
                path_to_transcript[wav_path] = norm_transcript
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_elizabeth(re_cache=False):
    root = "/mount/resources/speech/corpora/MAILabs_british_single_speaker_elizabeth"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        for el in os.listdir(root):
            if os.path.isdir(os.path.join(root, el)):
                with open(os.path.join(root, el, "metadata.csv"), "r", encoding="utf8") as file:
                    lookup = file.read()
                for line in lookup.split("\n"):
                    if line.strip() != "":
                        norm_transcript = line.split("|")[2]
                        wav_path = os.path.join(root, el, "wavs", line.split("|")[0] + ".wav")
                        if os.path.exists(wav_path):
                            path_to_transcript[wav_path] = norm_transcript
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_nancy(re_cache=False):
    root = "/mount/resources/speech/corpora/NancyKrebs"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(os.path.join(root, "metadata.csv"), "r", encoding="utf8") as file:
            lookup = file.read()
        for line in lookup.split("\n"):
            if line.strip() != "":
                norm_transcript = line.split("|")[1]
                wav_path = os.path.join(root, "wav", line.split("|")[0] + ".wav")
                if os.path.exists(wav_path):
                    path_to_transcript[wav_path] = norm_transcript
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_integration_test(re_cache=True):
    root = "/mount/resources/speech/corpora/NancyKrebs"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(os.path.join(root, "metadata.csv"), "r", encoding="utf8") as file:
            lookup = file.read()
        for line in lookup.split("\n")[:500]:
            if line.strip() != "":
                norm_transcript = line.split("|")[1]
                wav_path = os.path.join(root, "wav", line.split("|")[0] + ".wav")
                if os.path.exists(wav_path):
                    path_to_transcript[wav_path] = norm_transcript
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_CREMA_D(re_cache=False):
    root = "/mount/resources/speech/corpora/CREMA_D/"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        identifier_to_sent = {"IEO": "It's eleven o'clock.",
                              "TIE": "That is exactly what happened.",
                              "IOM": "I'm on my way to the meeting.",
                              "IWW": "I wonder what this is about.",
                              "TAI": "The airplane is almost full.",
                              "MTI": "Maybe tomorrow it will be cold.",
                              "IWL": "I would like a new alarm clock.",
                              "ITH": "I think, I have a doctor's appointment.",
                              "DFA": "Don't forget a jacket.",
                              "ITS": "I think, I've seen this before.",
                              "TSI": "The surface is slick.",
                              "WSI": "We'll stop in a couple of minutes."}
        path_to_transcript = dict()
        for file in os.listdir(root):
            if file.endswith(".wav"):
                path_to_transcript[root + file] = identifier_to_sent[file.split("_")[1]]
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_EmoV_DB(re_cache=False):
    root = "/mount/resources/speech/corpora/EmoV_DB/"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(os.path.join(root, "labels.txt"), "r", encoding="utf8") as file:
            lookup = file.read()
        identifier_to_sent = dict()
        for line in lookup.split("\n"):
            if line.strip() != "":
                identifier_to_sent[line.split()[0]] = " ".join(line.split()[1:])
        for file in os.listdir(root):
            if file.endswith(".wav"):
                path_to_transcript[root + file] = identifier_to_sent[file[-14:-10]]
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_ryanspeech(re_cache=False):
    root = "/mount/resources/speech/corpora/RyanSpeech"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript_dict = dict()
        with open(root + "/metadata.csv", mode="r", encoding="utf8") as f:
            transcripts = f.read().split("\n")
        for transcript in transcripts:
            if transcript.strip() != "":
                parsed_line = transcript.split("|")
                audio_file = f"{root}/wavs/{parsed_line[0]}.wav"
                path_to_transcript_dict[audio_file] = parsed_line[2]
        torch.save(path_to_transcript_dict, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_RAVDESS(re_cache=False):
    root = "/mount/resources/speech/corpora/RAVDESS"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript_dict = dict()
        for speaker_dir in os.listdir(root):
            for audio_file in os.listdir(os.path.join(root, speaker_dir)):
                if audio_file.split("-")[4] == "01":
                    path_to_transcript_dict[os.path.join(root, speaker_dir, audio_file)] = "Kids are talking by the door."
                else:
                    path_to_transcript_dict[os.path.join(root, speaker_dir, audio_file)] = "Dogs are sitting by the door."
        torch.save(path_to_transcript_dict, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_ESDS(re_cache=False):
    root = "/mount/resources/speech/corpora/Emotional_Speech_Dataset_Singapore"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript_dict = dict()
        for speaker_dir in os.listdir(root):
            if speaker_dir.startswith("00"):
                if int(speaker_dir) > 10:
                    with open(f"{root}/{speaker_dir}/fixed_unicode.txt", mode="r", encoding="utf8") as f:
                        transcripts = f.read()
                    for line in transcripts.replace("\n\n", "\n").replace(",", ", ").split("\n"):
                        if line.strip() != "":
                            filename, text, emo_dir = line.split("\t")
                            filename = speaker_dir + "_" + filename.split("_")[1]
                            path_to_transcript_dict[f"{root}/{speaker_dir}/{emo_dir}/{filename}.wav"] = text
        torch.save(path_to_transcript_dict, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_nvidia_hifitts(re_cache=False):
    root = "/mount/resources/speech/corpora/hi_fi_tts_v0"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        transcripts = list()
        import json
        for jpath in [f"{root}/6097_manifest_clean_dev.json",
                      f"{root}/6097_manifest_clean_test.json",
                      f"{root}/6097_manifest_clean_train.json",
                      f"{root}/9017_manifest_clean_dev.json",
                      f"{root}/9017_manifest_clean_test.json",
                      f"{root}/9017_manifest_clean_train.json",
                      f"{root}/92_manifest_clean_dev.json",
                      f"{root}/92_manifest_clean_test.json",
                      f"{root}/92_manifest_clean_train.json"]:
            with open(jpath, encoding='utf-8', mode='r') as jfile:
                for line in jfile.read().split("\n"):
                    if line.strip() != "":
                        transcripts.append(json.loads(line))
        for transcript in transcripts:
            path = transcript["audio_filepath"]
            norm_text = transcript["text_normalized"]
            path_to_transcript[f"{root}/{path}"] = norm_text
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_blizzard_2013(re_cache=False):
    root = "/mount/resources/speech/corpora/Blizzard2013/train/segmented/"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(root + "prompts.gui", encoding="utf8") as f:
            transcriptions = f.read()
        blocks = transcriptions.split("||\n")
        for block in blocks:
            trans_lines = block.split("\n")
            if trans_lines[0].strip() != "":
                transcript = trans_lines[1].replace("@", "").replace("#", ",").replace("|", "").replace(";", ",").replace(
                    ":", ",").replace(" 's", "'s").replace(", ,", ",").replace("  ", " ").replace(" ,", ",").replace(" .",
                                                                                                                     ".").replace(
                    " ?", "?").replace(" !", "!").rstrip(" ,")
                path_to_transcript[root + "wavn/" + trans_lines[0] + ".wav"] = transcript
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_vctk(re_cache=False):
    root = "/mount/resources/speech/corpora/VCTK"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        for transcript_dir in os.listdir("/mount/resources/speech/corpora/VCTK/txt"):
            for transcript_file in os.listdir(f"/mount/resources/speech/corpora/VCTK/txt/{transcript_dir}"):
                if transcript_file.endswith(".txt"):
                    with open(f"/mount/resources/speech/corpora/VCTK/txt/{transcript_dir}/" + transcript_file, 'r',
                              encoding='utf8') as tf:
                        transcript = tf.read()
                    wav_path = f"/mount/resources/speech/corpora/VCTK/wav48_silence_trimmed/{transcript_dir}/" + transcript_file.rstrip(
                        ".txt") + "_mic2.flac"
                    if os.path.exists(wav_path):
                        path_to_transcript[wav_path] = transcript
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_libritts_all_clean(re_cache=False):
    root = "/mount/resources/speech/corpora/LibriTTS_R/"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_train = "/mount/resources/speech/corpora/LibriTTS_R/"  # using all files from the "clean" subsets from LibriTTS-R https://arxiv.org/abs/2305.18802
        path_to_transcript = dict()
        for speaker in os.listdir(path_train):
            for chapter in os.listdir(os.path.join(path_train, speaker)):
                for file in os.listdir(os.path.join(path_train, speaker, chapter)):
                    if file.endswith("normalized.txt"):
                        with open(os.path.join(path_train, speaker, chapter, file), 'r', encoding='utf8') as tf:
                            transcript = tf.read()
                        wav_file = file.split(".")[0] + ".wav"
                        path_to_transcript[os.path.join(path_train, speaker, chapter, wav_file)] = transcript
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_libritts_other500(re_cache=False):
    root = "/mount/resources/asr-data/LibriTTS/train-other-500"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_train = "/mount/resources/asr-data/LibriTTS/train-other-500"
        path_to_transcript = dict()
        for speaker in os.listdir(path_train):
            for chapter in os.listdir(os.path.join(path_train, speaker)):
                for file in os.listdir(os.path.join(path_train, speaker, chapter)):
                    if file.endswith("normalized.txt"):
                        with open(os.path.join(path_train, speaker, chapter, file), 'r', encoding='utf8') as tf:
                            transcript = tf.read()
                        wav_file = file.split(".")[0] + ".wav"
                        path_to_transcript[os.path.join(path_train, speaker, chapter, wav_file)] = transcript
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_ljspeech(re_cache=False):
    root = "/mount/resources/speech/corpora/LJSpeech/"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        for transcript_file in os.listdir("/mount/resources/speech/corpora/LJSpeech/16kHz/txt"):
            with open("/mount/resources/speech/corpora/LJSpeech/16kHz/txt/" + transcript_file, 'r', encoding='utf8') as tf:
                transcript = tf.read()
            wav_path = "/mount/resources/speech/corpora/LJSpeech/16kHz/wav/" + transcript_file.rstrip(".txt") + ".wav"
            path_to_transcript[wav_path] = transcript
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_jenny(re_cache=False):
    """
    https://www.kaggle.com/datasets/noml4u/jenny-tts-dataset
    https://github.com/dioco-group/jenny-tts-dataset

    Dataset of Speaker Jenny (Dioco) with an Irish accent
    """
    root = "/mount/resources/speech/corpora/Jenny/"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open("/mount/resources/speech/corpora/Jenny/metadata.csv", encoding="utf8") as f:
            transcriptions = f.read()
        trans_lines = transcriptions.split("\n")
        for line in trans_lines:
            if line.strip() != "":
                path_to_transcript["/mount/resources/speech/corpora/Jenny/" + line.split("|")[0] + "_silence.flac"] = line.split("|")[1]
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


# GERMAN

def build_path_to_transcript_dict_mls_german(re_cache=False):
    lang = "german"
    root = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = build_path_to_transcript_dict_multi_ling_librispeech_template(root=root)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_karlsson(re_cache=False):
    root = "/mount/resources/speech/corpora/HUI_German/Karlsson"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = build_path_to_transcript_dict_hui_template(root=root)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_eva(re_cache=False):
    root = "/mount/resources/speech/corpora/HUI_German/Eva"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = build_path_to_transcript_dict_hui_template(root=root)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_bernd(re_cache=False):
    root = "/mount/resources/speech/corpora/HUI_German/Bernd"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = build_path_to_transcript_dict_hui_template(root=root)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_friedrich(re_cache=False):
    root = "/mount/resources/speech/corpora/HUI_German/Friedrich"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = build_path_to_transcript_dict_hui_template(root=root)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_hokus(re_cache=False):
    root = "/mount/resources/speech/corpora/HUI_German/Hokus"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = build_path_to_transcript_dict_hui_template(root=root)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_hui_others(re_cache=False):
    root = "/mount/resources/speech/corpora/HUI_German/others"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        pttd = dict()
        for speaker in os.listdir(root):
            pttd.update(build_path_to_transcript_dict_hui_template(root=f"{root}/{speaker}"))
        torch.save(pttd, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_thorsten_neutral(re_cache=False):
    root = "/mount/resources/speech/corpora/ThorstenDatasets/thorsten-de_v03"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(root + "/metadata_train.csv", encoding="utf8") as f:
            transcriptions = f.read()
        with open(root + "/metadata_val.csv", encoding="utf8") as f:
            transcriptions += "\n" + f.read()
        trans_lines = transcriptions.split("\n")
        for line in trans_lines:
            if line.strip() != "":
                path_to_transcript[root + "/wavs/" + line.split("|")[0] + ".wav"] = \
                    line.split("|")[1]
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_thorsten_2022_10(re_cache=False):
    root = "/mount/resources/speech/corpora/ThorstenDatasets/ThorstenVoice-Dataset_2022.10"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(root + "/metadata_train.csv", encoding="utf8") as f:
            transcriptions = f.read()
        with open(root + "/metadata_dev.csv", encoding="utf8") as f:
            transcriptions += "\n" + f.read()
        with open(root + "/metadata_test.csv", encoding="utf8") as f:
            transcriptions += "\n" + f.read()
        trans_lines = transcriptions.split("\n")
        for line in trans_lines:
            if line.strip() != "":
                path_to_transcript[root + "/wavs/" + line.split("|")[0] + ".wav"] = \
                    line.split("|")[1]
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_thorsten_emotional(re_cache=False):
    root = "/mount/resources/speech/corpora/ThorstenDatasets/thorsten-emotional_v02"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(root + "/thorsten-emotional-metadata.csv", encoding="utf8") as f:
            transcriptions = f.read()
        trans_lines = transcriptions.split("\n")
        for line in trans_lines:
            if line.strip() != "":
                path_to_transcript[root + "/amused/" + line.split("|")[0] + ".wav"] = line.split("|")[1]
                path_to_transcript[root + "/angry/" + line.split("|")[0] + ".wav"] = line.split("|")[1]
                path_to_transcript[root + "/disgusted/" + line.split("|")[0] + ".wav"] = line.split("|")[1]
                path_to_transcript[root + "/neutral/" + line.split("|")[0] + ".wav"] = line.split("|")[1]
                path_to_transcript[root + "/sleepy/" + line.split("|")[0] + ".wav"] = line.split("|")[1]
                path_to_transcript[root + "/surprised/" + line.split("|")[0] + ".wav"] = line.split("|")[1]
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


# FRENCH

def build_path_to_transcript_dict_mls_french(re_cache=False):
    lang = "french"
    root = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = build_path_to_transcript_dict_multi_ling_librispeech_template(root=root)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_blizzard2023_ad_silence_removed(re_cache=False):
    root = "/mount/resources/speech/corpora/Blizzard2023/AD_silence_removed"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(os.path.join(root, "transcript.tsv"), "r", encoding="utf8") as file:
            lookup = file.read()
        for line in lookup.split("\n"):
            if line.strip() != "":
                norm_transcript = line.split("\t")[1]
                wav_path = os.path.join(root, line.split("\t")[0].split("/")[-1])
                if os.path.exists(wav_path):
                    path_to_transcript[wav_path] = norm_transcript.replace("§", "").replace("#", "").replace("~", "").replace(" »", '"').replace("« ", '"').replace("»", '"').replace("«", '"')
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_blizzard2023_neb_silence_removed(re_cache=False):
    root = "/mount/resources/speech/corpora/Blizzard2023/NEB_silence_removed"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(os.path.join(root, "transcript.tsv"), "r", encoding="utf8") as file:
            lookup = file.read()
        for line in lookup.split("\n"):
            if line.strip() != "":
                norm_transcript = line.split("\t")[1]
                wav_path = os.path.join(root, line.split("\t")[0].split("/")[-1])
                if os.path.exists(wav_path):
                    path_to_transcript[wav_path] = norm_transcript.replace("§", "").replace("#", "").replace("~", "").replace(" »", '"').replace("« ", '"').replace("»", '"').replace("«", '"')
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_blizzard2023_neb_e_silence_removed(re_cache=False):
    root = "/mount/resources/speech/corpora/Blizzard2023/enhanced_NEB_subset_silence_removed"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(os.path.join(root, "transcript.tsv"), "r", encoding="utf8") as file:
            lookup = file.read()
        for line in lookup.split("\n"):
            if line.strip() != "":
                norm_transcript = line.split("\t")[1]
                wav_path = os.path.join(root, line.split("\t")[0].split("/")[-1])
                if os.path.exists(wav_path):
                    path_to_transcript[wav_path] = norm_transcript.replace("§", "").replace("#", "").replace("~", "").replace(" »", '"').replace("« ", '"').replace("»", '"').replace("«", '"')
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_synpaflex_norm_subset(re_cache=False):
    """
    Contributed by https://github.com/tomschelsen
    """
    root = "/mount/resources/speech/corpora/synpaflex-corpus/5/v0.1/"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        for text_path in glob.iglob(os.path.join(root, "**/*_norm.txt"), recursive=True):
            with open(text_path, "r", encoding="utf8") as file:
                norm_transcript = file.read()
            path_obj = Path(text_path)
            wav_path = str((path_obj.parent.parent / path_obj.name[:-9]).with_suffix(".wav"))
            if Path(wav_path).exists():
                path_to_transcript[wav_path] = norm_transcript
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_siwis_subset(re_cache=False):
    """
    Contributed by https://github.com/tomschelsen
    """
    root = "/mount/resources/speech/corpora/SiwisFrenchSpeechSynthesisDatabase/"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        # part4 and part5 are not segmented
        sub_dirs = ["part1", "part2", "part3"]
        path_to_transcript = dict()
        for sd in sub_dirs:
            for text_path in glob.iglob(os.path.join(root, "text", sd, "*.txt")):
                with open(text_path, "r", encoding="utf8") as file:
                    norm_transcript = file.read()
                path_obj = Path(text_path)
                wav_path = str((path_obj.parent.parent.parent / "wavs" / sd / path_obj.stem).with_suffix(".wav"))
                if Path(wav_path).exists():
                    path_to_transcript[wav_path] = norm_transcript
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_css10fr(re_cache=False):
    language = "french"
    root = f"/mount/resources/speech/corpora/CSS10/{language}"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(f"/mount/resources/speech/corpora/CSS10/{language}/transcript.txt", encoding="utf8") as f:
            transcriptions = f.read()
        trans_lines = transcriptions.split("\n")
        for line in trans_lines:
            if line.strip() != "":
                path_to_transcript[f"/mount/resources/speech/corpora/CSS10/{language}/{line.split('|')[0]}"] = \
                    line.split("|")[2]
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


# SPANISH

def build_path_to_transcript_dict_mls_spanish(re_cache=False):
    lang = "spanish"
    root = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = build_path_to_transcript_dict_multi_ling_librispeech_template(root=root)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_css10es(re_cache=False):
    language = "spanish"
    root = f"/mount/resources/speech/corpora/CSS10/{language}"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(f"/mount/resources/speech/corpora/CSS10/{language}/transcript.txt", encoding="utf8") as f:
            transcriptions = f.read()
        trans_lines = transcriptions.split("\n")
        for line in trans_lines:
            if line.strip() != "":
                path_to_transcript[f"/mount/resources/speech/corpora/CSS10/{language}/{line.split('|')[0]}"] = \
                    line.split("|")[2]
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_spanish_blizzard_train(re_cache=False):
    root = "/mount/resources/speech/corpora/Blizzard2021/spanish_blizzard_release_2021_v2/hub"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(os.path.join(root, "train_text.txt"), "r", encoding="utf8") as file:
            lookup = file.read()
        for line in lookup.split("\n"):
            if line.strip() != "":
                norm_transcript = line.split("\t")[1]
                wav_path = os.path.join(root, "train_wav", line.split("\t")[0] + ".wav")
                if os.path.exists(wav_path):
                    path_to_transcript[wav_path] = norm_transcript
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


# PORTUGUESE

def build_path_to_transcript_dict_mls_portuguese(re_cache=False):
    lang = "portuguese"
    root = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = build_path_to_transcript_dict_multi_ling_librispeech_template(root=root)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


# POLISH

def build_path_to_transcript_dict_mls_polish(re_cache=False):
    lang = "polish"
    root = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = build_path_to_transcript_dict_multi_ling_librispeech_template(root=root)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


# ITALIAN

def build_path_to_transcript_dict_mls_italian(re_cache=False):
    lang = "italian"
    root = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = build_path_to_transcript_dict_multi_ling_librispeech_template(root=root)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


# DUTCH

def build_path_to_transcript_dict_mls_dutch(re_cache=False):
    lang = "dutch"
    root = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = build_path_to_transcript_dict_multi_ling_librispeech_template(root=root)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_css10nl(re_cache=False):
    language = "dutch"
    root = f"/mount/resources/speech/corpora/CSS10/{language}"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(f"/mount/resources/speech/corpora/CSS10/{language}/transcript.txt", encoding="utf8") as f:
            transcriptions = f.read()
        trans_lines = transcriptions.split("\n")
        for line in trans_lines:
            if line.strip() != "":
                path_to_transcript[f"/mount/resources/speech/corpora/CSS10/{language}/{line.split('|')[0]}"] = \
                    line.split("|")[2]
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


# GREEK

def build_path_to_transcript_dict_css10el(re_cache=False):
    language = "greek"
    root = f"/mount/resources/speech/corpora/CSS10/{language}"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(f"/mount/resources/speech/corpora/CSS10/{language}/transcript.txt", encoding="utf8") as f:
            transcriptions = f.read()
        trans_lines = transcriptions.split("\n")
        for line in trans_lines:
            if line.strip() != "":
                path_to_transcript[f"/mount/resources/speech/corpora/CSS10/{language}/{line.split('|')[0]}"] = \
                    line.split("|")[2]
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


# FINNISH

def build_path_to_transcript_dict_css10fi(re_cache=False):
    language = "finnish"
    root = f"/mount/resources/speech/corpora/CSS10/{language}"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(f"/mount/resources/speech/corpora/CSS10/{language}/transcript.txt", encoding="utf8") as f:
            transcriptions = f.read()
        trans_lines = transcriptions.split("\n")
        for line in trans_lines:
            if line.strip() != "":
                path_to_transcript[f"/mount/resources/speech/corpora/CSS10/{language}/{line.split('|')[0]}"] = \
                    line.split("|")[2]
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


# VIETNAMESE


def build_path_to_transcript_dict_VIVOS_viet(re_cache=False):
    root = "/mount/resources/speech/corpora/VIVOS_vietnamese/train"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript_dict = dict()
        with open(root + "/prompts.txt", mode="r", encoding="utf8") as f:
            transcripts = f.read().split("\n")
        for transcript in transcripts:
            if transcript.strip() != "":
                parsed_line = transcript.split(" ")
                audio_file = f"{root}/waves/{parsed_line[0][:10]}/{parsed_line[0]}.wav"
                path_to_transcript_dict[audio_file] = " ".join(parsed_line[1:]).lower()
        torch.save(path_to_transcript_dict, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_vietTTS(re_cache=False):
    root = "/mount/resources/speech/corpora/VietTTS"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(root + "/meta_data.tsv", encoding="utf8") as f:
            transcriptions = f.read()
        for line in transcriptions.split("\n"):
            if line.strip() != "":
                parsed_line = line.split(".wav")
                audio_path = parsed_line[0]
                transcript = parsed_line[1]
                path_to_transcript[os.path.join(root, audio_path + ".wav")] = transcript.strip()
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


# CHINESE

def build_path_to_transcript_dict_aishell3(re_cache=False):
    root = "/mount/resources/speech/corpora/aishell3/train"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript_dict = dict()
        with open(root + "/label_train-set.txt", mode="r", encoding="utf8") as f:
            transcripts = f.read().replace("$", "").replace("%", " ").split("\n")
        for transcript in transcripts:
            if transcript.strip() != "" and not transcript.startswith("#"):
                parsed_line = transcript.split("|")
                audio_file = f"{root}/wav/{parsed_line[0][:7]}/{parsed_line[0]}.wav"
                kanji = parsed_line[2]
                path_to_transcript_dict[audio_file] = kanji
        torch.save(path_to_transcript_dict, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_css10cmn(re_cache=False):
    language = "chinese"
    root = f"/mount/resources/speech/corpora/CSS10/{language}"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open("/mount/resources/speech/corpora/CSS10/chinese/transcript.txt", encoding="utf8") as f:
            transcriptions = f.read()
        trans_lines = transcriptions.split("\n")
        for line in trans_lines:
            if line.strip() != "":
                path_to_transcript["/mount/resources/speech/corpora/CSS10/chinese/" + line.split("|")[0]] = line.split("|")[2]
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


# RUSSIAN

def build_path_to_transcript_dict_css10ru(re_cache=False):
    language = "russian"
    root = f"/mount/resources/speech/corpora/CSS10/{language}"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(f"/mount/resources/speech/corpora/CSS10/{language}/transcript.txt", encoding="utf8") as f:
            transcriptions = f.read()
        trans_lines = transcriptions.split("\n")
        for line in trans_lines:
            if line.strip() != "":
                path_to_transcript[f"/mount/resources/speech/corpora/CSS10/{language}/{line.split('|')[0]}"] = \
                    line.split("|")[2]
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


# HUNGARIAN

def build_path_to_transcript_dict_css10hu(re_cache=False):
    language = "hungarian"
    root = f"/mount/resources/speech/corpora/CSS10/{language}"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        language = "hungarian"
        with open(f"/mount/resources/speech/corpora/CSS10/{language}/transcript.txt", encoding="utf8") as f:
            transcriptions = f.read()
        trans_lines = transcriptions.split("\n")
        for line in trans_lines:
            if line.strip() != "":
                path_to_transcript[f"/mount/resources/speech/corpora/CSS10/{language}/{line.split('|')[0]}"] = \
                    line.split("|")[2]
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


# OTHER

def build_file_list_singing_voice_audio_database(re_cache=False):
    root = "/mount/resources/speech/corpora/singing_voice_audio_dataset/monophonic"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        file_list = list()
        for corw in os.listdir(root):
            for singer in os.listdir(os.path.join(root, corw)):
                for audio in os.listdir(os.path.join(root, corw, singer)):
                    file_list.append(os.path.join(root, corw, singer, audio))
        torch.save(file_list, cache_path)
    return torch.load(cache_path)


def prepare_aligner_corpus(transcript_dict, corpus_dir, lang, device):
    if type(transcript_dict) != dict:
        transcript_dict = transcript_dict()
    print(corpus_dir)
    return transcript_dict


if __name__ == '__main__':
    pass
