import os


def build_path_to_transcript_dict_mls_italian():
    lang = "italian"
    root = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train"
    return build_path_to_transcript_dict_multi_ling_librispeech_template(root=root)


def build_path_to_transcript_dict_mls_french():
    lang = "french"
    root = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train"
    return build_path_to_transcript_dict_multi_ling_librispeech_template(root=root)


def build_path_to_transcript_dict_mls_dutch():
    lang = "dutch"
    root = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train"
    return build_path_to_transcript_dict_multi_ling_librispeech_template(root=root)


def build_path_to_transcript_dict_mls_polish():
    lang = "polish"
    root = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train"
    return build_path_to_transcript_dict_multi_ling_librispeech_template(root=root)


def build_path_to_transcript_dict_mls_spanish():
    lang = "spanish"
    root = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train"
    return build_path_to_transcript_dict_multi_ling_librispeech_template(root=root)


def build_path_to_transcript_dict_mls_portuguese():
    lang = "portuguese"
    root = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train"
    return build_path_to_transcript_dict_multi_ling_librispeech_template(root=root)


def build_path_to_transcript_dict_multi_ling_librispeech_template(root):
    """
    https://arxiv.org/abs/2012.03411
    """
    path_to_transcript = dict()
    with open(os.path.join(root, "transcripts.txt"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            norm_transcript = line.split("\t")[1]
            wav_folders = line.split("\t")[0].split("_")
            wav_path = os.path.join(root, "audio", wav_folders[0], wav_folders[1], line.split("\t")[0] + ".flac")
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = norm_transcript
            else:
                print(f"not found: {wav_path}")
    return path_to_transcript


def build_path_to_transcript_dict_karlsson():
    root = "/mount/resources/speech/corpora/HUI_German/Karlsson"
    return build_path_to_transcript_dict_hui_template(root=root)


def build_path_to_transcript_dict_eva():
    root = "/mount/resources/speech/corpora/HUI_German/Eva"
    return build_path_to_transcript_dict_hui_template(root=root)


def build_path_to_transcript_dict_bernd():
    root = "/mount/resources/speech/corpora/HUI_German/Bernd"
    return build_path_to_transcript_dict_hui_template(root=root)


def build_path_to_transcript_dict_friedrich():
    root = "/mount/resources/speech/corpora/HUI_German/Friedrich"
    return build_path_to_transcript_dict_hui_template(root=root)


def build_path_to_transcript_dict_hokus():
    root = "/mount/resources/speech/corpora/HUI_German/Hokus"
    return build_path_to_transcript_dict_hui_template(root=root)


def build_path_to_transcript_dict_hui_others():
    root = "/mount/resources/speech/corpora/HUI_German/others"
    pttd = dict()
    for speaker in os.listdir(root):
        pttd.update(build_path_to_transcript_dict_hui_template(root=f"{root}/{speaker}"))
    return pttd


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


def build_path_to_transcript_dict_fluxsing():
    root = "/mount/resources/speech/corpora/FluxSing"
    path_to_transcript = dict()
    with open(os.path.join(root, "metadata.csv"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            norm_transcript = line.split("|")[2]
            wav_path = os.path.join(root, line.split("|")[0])
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = norm_transcript
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


def build_path_to_transcript_dict_libritts_all_clean():
    path_train = "/mount/resources/speech/corpora/LibriTTS/all-clean"
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


def build_path_to_transcript_dict_libritts_other500():
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
    return path_to_transcript


def build_path_to_transcript_dict_libritts_asr_other500(label_file):
    with open(label_file, encoding="utf8", mode="r") as f:
        labels = f.read()
    audio_handle_to_transcript = dict()
    for line in labels.split("\n"):
        if line.strip() == "":
            continue
        audio_handle_to_transcript[line.split()[0]] = line.lstrip(f"{line.split()[0]} ")
    path_train = "/mount/resources/asr-data/LibriTTS/train-other-500"
    path_to_transcript = dict()
    for speaker in os.listdir(path_train):
        for chapter in os.listdir(os.path.join(path_train, speaker)):
            for file in os.listdir(os.path.join(path_train, speaker, chapter)):
                if file.endswith(".wav"):
                    try:
                        path_to_transcript[os.path.join(path_train, speaker, chapter, file)] = audio_handle_to_transcript[file.split(".")[0]]
                    except KeyError:
                        print(f"Problem with {file}, no transcription found!")
    return path_to_transcript


def build_path_to_transcript_dict_libritts_asr(label_file):
    with open(label_file, encoding="utf8", mode="r") as f:
        labels = f.read()
    audio_handle_to_transcript = dict()
    for line in labels.split("\n"):
        if line.strip() == "":
            continue
        audio_handle_to_transcript[line.split()[0]] = line.lstrip(f"{line.split()[0]} ")
    path_train = "/mount/resources/speech/corpora/LibriTTS/train-clean-100"
    path_to_transcript = dict()
    for speaker in os.listdir(path_train):
        for chapter in os.listdir(os.path.join(path_train, speaker)):
            for file in os.listdir(os.path.join(path_train, speaker, chapter)):
                if file.endswith(".wav"):
                    try:
                        path_to_transcript[os.path.join(path_train, speaker, chapter, file)] = audio_handle_to_transcript[file.split(".")[0]]
                    except KeyError:
                        print(f"Problem with {file}, no transcription found!")
    return path_to_transcript


def build_path_to_transcript_dict_libritts_asr_out():
    return build_path_to_transcript_dict_libritts_asr("/mount/arbeitsdaten45/projekte/asr-4/denisopl/tmp/libritts_train_600_tts-bpe100.txt")


def build_path_to_transcript_dict_libritts_asr_phn():
    return build_path_to_transcript_dict_libritts_asr("/mount/arbeitsdaten45/projekte/asr-4/denisopl/tmp/libritts_train_600_tts-phn-bpe100.txt")


def build_path_to_transcript_dict_libritts_asr_out_500():
    return build_path_to_transcript_dict_libritts_asr_other500("/mount/arbeitsdaten45/projekte/asr-4/denisopl/tmp/libritts_train_600_tts-bpe100.txt")


def build_path_to_transcript_dict_libritts_asr_phn_500():
    return build_path_to_transcript_dict_libritts_asr_other500("/mount/arbeitsdaten45/projekte/asr-4/denisopl/tmp/libritts_train_600_tts-phn-bpe100.txt")


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


def build_path_to_transcript_dict_spanish_blizzard_train():
    root = "/mount/resources/speech/corpora/Blizzard2021/spanish_blizzard_release_2021_v2/hub"
    path_to_transcript = dict()
    with open(os.path.join(root, "train_text.txt"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            norm_transcript = line.split("\t")[1]
            wav_path = os.path.join(root, "train_wav", line.split("\t")[0] + ".wav")
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = norm_transcript
    return path_to_transcript


def build_path_to_transcript_dict_3xljspeech():
    path_to_transcript = dict()
    for transcript_file in os.listdir("/mount/arbeitsdaten/synthesis/attention_projects/LJSpeech_3xlong_stripped/txt_long"):
        with open("/mount/arbeitsdaten/synthesis/attention_projects/LJSpeech_3xlong_stripped/txt_long/" + transcript_file, 'r', encoding='utf8') as tf:
            transcript = tf.read()
        wav_path = "/mount/arbeitsdaten/synthesis/attention_projects/LJSpeech_3xlong_stripped/wav_long/" + transcript_file.rstrip(".txt") + ".wav"
        path_to_transcript[wav_path] = transcript
    return path_to_transcript
