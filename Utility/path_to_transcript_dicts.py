import os


def build_path_to_transcript_dict_karlsson():
    """
    https://arxiv.org/abs/2106.06309
    """
    root = "/mount/resources/speech/corpora/Karlsson"
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


def build_path_to_transcript_dict_eva():
    """
    https://arxiv.org/abs/2106.06309
    """
    root = "/mount/resources/speech/corpora/Eva_K"
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


def build_path_to_transcript_dict_elizabeth():
    root = "/mount/resources/speech/corpora/MAILabs_british_single_speaker_elizabeth"
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
    return path_to_transcript


def build_path_to_transcript_dict_nancy():
    root = "/mount/resources/speech/corpora/NancyKrebs"
    path_to_transcript = dict()
    with open(os.path.join(root, "metadata.csv"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            norm_transcript = line.split("|")[1]
            wav_path = os.path.join(root, "wav", line.split("|")[0] + ".wav")
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = norm_transcript
    return path_to_transcript


def build_path_to_transcript_dict_hokuspokus():
    path_to_transcript = dict()
    for transcript_file in os.listdir("/mount/resources/speech/corpora/LibriVox.Hokuspokus/txt"):
        if transcript_file.endswith(".txt"):
            with open("/mount/resources/speech/corpora/LibriVox.Hokuspokus/txt/" + transcript_file, 'r', encoding='utf8') as tf:
                transcript = tf.read()
            wav_path = "/mount/resources/speech/corpora/LibriVox.Hokuspokus/wav/" + transcript_file.rstrip(".txt") + ".wav"
            path_to_transcript[wav_path] = transcript
    return path_to_transcript


def build_path_to_transcript_dict_vctk():
    path_to_transcript = dict()
    for transcript_dir in os.listdir("/mount/resources/speech/corpora/VCTK/txt"):
        for transcript_file in os.listdir(f"/mount/resources/speech/corpora/VCTK/txt/{transcript_dir}"):
            if transcript_file.endswith(".txt"):
                with open(f"/mount/resources/speech/corpora/VCTK/txt/{transcript_dir}/" + transcript_file, 'r', encoding='utf8') as tf:
                    transcript = tf.read()
                wav_path = f"/mount/resources/speech/corpora/VCTK/wav48_silence_trimmed/{transcript_dir}/" + transcript_file.rstrip(".txt") + "_mic2.flac"
                if os.path.exists(wav_path):
                    path_to_transcript[wav_path] = transcript
    return path_to_transcript


def build_path_to_transcript_dict_libritts():
    path_train = "/mount/resources/speech/corpora/LibriTTS/train-clean-100"
    path_to_transcript = dict()
    for speaker in os.listdir(path_train):
        for chapter in os.listdir(os.path.join(path_train, speaker)):
            for file in os.listdir(os.path.join(path_train, speaker, chapter)):
                if file.endswith("normalized.txt"):
                    with open(os.path.join(path_train, speaker, chapter, file), 'r', encoding='utf8') as tf:
                        transcript = tf.read()
                    wav_file = file.split(".")[0] + ".wav"
                    path_to_transcript[os.path.join(path_train, speaker, chapter, wav_file)] = transcript
    return path_to_transcript


def build_path_to_transcript_dict_ljspeech():
    path_to_transcript = dict()
    for transcript_file in os.listdir("/mount/resources/speech/corpora/LJSpeech/16kHz/txt"):
        with open("/mount/resources/speech/corpora/LJSpeech/16kHz/txt/" + transcript_file, 'r', encoding='utf8') as tf:
            transcript = tf.read()
        wav_path = "/mount/resources/speech/corpora/LJSpeech/16kHz/wav/" + transcript_file.rstrip(".txt") + ".wav"
        path_to_transcript[wav_path] = transcript
    return path_to_transcript


def build_path_to_transcript_dict_att_hack():
    path_to_transcript = dict()
    for transcript_file in os.listdir("/mount/resources/speech/corpora/FrenchExpressive/txt"):
        if transcript_file.endswith(".txt"):
            with open("/mount/resources/speech/corpora/FrenchExpressive/txt/" + transcript_file, 'r', encoding='utf8') as tf:
                transcript = tf.read()
            wav_path = "/mount/resources/speech/corpora/FrenchExpressive/wav/" + transcript_file.rstrip(".txt") + ".wav"
            path_to_transcript[wav_path] = transcript
    return path_to_transcript


def build_path_to_transcript_dict_css10de():
    path_to_transcript = dict()
    with open("/mount/resources/speech/corpora/CSS10/german/transcript.txt", encoding="utf8") as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            path_to_transcript["/mount/resources/speech/corpora/CSS10/german/" + line.split("|")[0]] = line.split("|")[2]
    return path_to_transcript


def build_path_to_transcript_dict_thorsten():
    path_to_transcript = dict()
    with open("/mount/resources/speech/corpora/Thorsten_DE/metadata_shuf.csv", encoding="utf8") as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            path_to_transcript["/mount/resources/speech/corpora/Thorsten_DE/wavs/" + line.split("|")[0] + ".wav"] = line.split("|")[1]
    return path_to_transcript


def build_path_to_transcript_dict_css10el():
    path_to_transcript = dict()
    language = "greek"
    with open(f"/mount/resources/speech/corpora/CSS10/{language}/transcript.txt", encoding="utf8") as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            path_to_transcript[f"/mount/resources/speech/corpora/CSS10/{language}/{line.split('|')[0]}"] = line.split("|")[2]
    return path_to_transcript


def build_path_to_transcript_dict_css10nl():
    path_to_transcript = dict()
    language = "dutch"
    with open(f"/mount/resources/speech/corpora/CSS10/{language}/transcript.txt", encoding="utf8") as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            path_to_transcript[f"/mount/resources/speech/corpora/CSS10/{language}/{line.split('|')[0]}"] = line.split("|")[2]
    return path_to_transcript


def build_path_to_transcript_dict_css10fi():
    path_to_transcript = dict()
    language = "finnish"
    with open(f"/mount/resources/speech/corpora/CSS10/{language}/transcript.txt", encoding="utf8") as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            path_to_transcript[f"/mount/resources/speech/corpora/CSS10/{language}/{line.split('|')[0]}"] = line.split("|")[2]
    return path_to_transcript


def build_path_to_transcript_dict_css10ru():
    path_to_transcript = dict()
    language = "russian"
    with open(f"/mount/resources/speech/corpora/CSS10/{language}/transcript.txt", encoding="utf8") as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            path_to_transcript[f"/mount/resources/speech/corpora/CSS10/{language}/{line.split('|')[0]}"] = line.split("|")[2]
    return path_to_transcript


def build_path_to_transcript_dict_css10hu():
    path_to_transcript = dict()
    language = "hungarian"
    with open(f"/mount/resources/speech/corpora/CSS10/{language}/transcript.txt", encoding="utf8") as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            path_to_transcript[f"/mount/resources/speech/corpora/CSS10/{language}/{line.split('|')[0]}"] = line.split("|")[2]
    return path_to_transcript


def build_path_to_transcript_dict_css10es():
    path_to_transcript = dict()
    language = "spanish"
    with open(f"/mount/resources/speech/corpora/CSS10/{language}/transcript.txt", encoding="utf8") as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            path_to_transcript[f"/mount/resources/speech/corpora/CSS10/{language}/{line.split('|')[0]}"] = line.split("|")[2]
    return path_to_transcript


def build_path_to_transcript_dict_css10fr():
    path_to_transcript = dict()
    language = "french"
    with open(f"/mount/resources/speech/corpora/CSS10/{language}/transcript.txt", encoding="utf8") as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            path_to_transcript[f"/mount/resources/speech/corpora/CSS10/{language}/{line.split('|')[0]}"] = line.split("|")[2]
    return path_to_transcript


def build_path_to_transcript_dict_nvidia_hifitts():
    path_to_transcript = dict()
    transcripts = list()
    root = "/mount/resources/speech/corpora/hi_fi_tts_v0"

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

    return path_to_transcript
