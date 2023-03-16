from Utility.path_to_transcript_dicts import *

def get_file_list_aridialect():
    path_to_transcript_dict = build_path_to_transcript_dict_aridialect()
    #print(path_to_transcript_dict.keys())
    return list(path_to_transcript_dict.keys())
    
def get_file_list_aridialect_48kHz():
    path_to_transcript_dict = build_path_to_transcript_dict_aridialect_48kHz()
    #print(path_to_transcript_dict.keys())
    return list(path_to_transcript_dict.keys())
    
def get_file_list_vctk():
    path_to_transcript_dict = build_path_to_transcript_dict_vctk()
    return list(path_to_transcript_dict.keys())


def get_file_list_fluxsing():
    path_to_transcript_dict = build_path_to_transcript_dict_fluxsing()
    return list(path_to_transcript_dict.keys())


def get_file_list_hui_karlsson():
    path_to_transcript_dict = build_path_to_transcript_dict_karlsson()
    return list(path_to_transcript_dict.keys())


def get_file_list_hui_eva():
    path_to_transcript_dict = build_path_to_transcript_dict_eva()
    return list(path_to_transcript_dict.keys())


def get_file_list_hui_friedrich():
    path_to_transcript_dict = build_path_to_transcript_dict_friedrich()
    return list(path_to_transcript_dict.keys())


def get_file_list_hui_bernd():
    path_to_transcript_dict = build_path_to_transcript_dict_bernd()
    return list(path_to_transcript_dict.keys())


def get_file_list_hui_hokus():
    path_to_transcript_dict = build_path_to_transcript_dict_hokus()
    return list(path_to_transcript_dict.keys())


def get_file_list_hui_others():
    path_to_transcript_dict = build_path_to_transcript_dict_hui_others()
    return list(path_to_transcript_dict.keys())


def get_file_list_elizabeth():
    root = "/mount/resources/speech/corpora/MAILabs_british_single_speaker_elizabeth"
    file_list = list()
    for el in os.listdir(root):
        if os.path.isdir(os.path.join(root, el)):
            with open(os.path.join(root, el, "metadata.csv"), "r", encoding="utf8") as file:
                lookup = file.read()
            for line in lookup.split("\n"):
                if line.strip() != "":
                    wav_path = os.path.join(root, el, "wavs", line.split("|")[0] + ".wav")
                    if os.path.exists(wav_path):
                        file_list.append(wav_path)
    return file_list


def get_file_list_nancy():
    file_list = list()
    for wav_file in os.listdir("/mount/resources/speech/corpora/NancyKrebs/wav"):
        if ".wav" in wav_file:
            file_list.append("/mount/resources/speech/corpora/NancyKrebs/wav/" + wav_file)
    return file_list


def get_file_list_libritts():
    path_train = "/mount/resources/speech/corpora/LibriTTS/all_clean"
    file_list = list()
    for speaker in os.listdir(path_train):
        for chapter in os.listdir(os.path.join(path_train, speaker)):
            for file in os.listdir(os.path.join(path_train, speaker, chapter)):
                if file.endswith(".wav"):
                    file_list.append(os.path.join(path_train, speaker, chapter, file))
    return file_list


def get_file_list_ljspeech():
    file_list = list()
    for wav_file in os.listdir("/mount/resources/speech/corpora/LJSpeech/16kHz/wav"):
        if ".wav" in wav_file:
            file_list.append("/mount/resources/speech/corpora/LJSpeech/16kHz/wav/" + wav_file)
    return file_list


def get_file_list_css10de():
    file_list = list()
    language = "german"
    for book in os.listdir(os.path.join("/mount/resources/speech/corpora/CSS10", language)):
        if os.path.isdir(os.path.join("/mount/resources/speech/corpora/CSS10", language, book)):
            for audio in os.listdir(os.path.join("/mount/resources/speech/corpora/CSS10", language, book)):
                if audio.endswith(".wav"):
                    file_list.append(os.path.join("/mount/resources/speech/corpora/CSS10", language, book, audio))
    return file_list


def get_file_list_css10gr():
    file_list = list()
    language = "greek"
    for book in os.listdir(os.path.join("/mount/resources/speech/corpora/CSS10", language)):
        if os.path.isdir(os.path.join("/mount/resources/speech/corpora/CSS10", language, book)):
            for audio in os.listdir(os.path.join("/mount/resources/speech/corpora/CSS10", language, book)):
                if audio.endswith(".wav"):
                    file_list.append(os.path.join("/mount/resources/speech/corpora/CSS10", language, book, audio))
    return file_list


def get_file_list_css10es():
    file_list = list()
    language = "spanish"
    for book in os.listdir(os.path.join("/mount/resources/speech/corpora/CSS10", language)):
        if os.path.isdir(os.path.join("/mount/resources/speech/corpora/CSS10", language, book)):
            for audio in os.listdir(os.path.join("/mount/resources/speech/corpora/CSS10", language, book)):
                if audio.endswith(".wav"):
                    file_list.append(os.path.join("/mount/resources/speech/corpora/CSS10", language, book, audio))
    return file_list


def get_file_list_css10fi():
    file_list = list()
    language = "finnish"
    for book in os.listdir(os.path.join("/mount/resources/speech/corpora/CSS10", language)):
        if os.path.isdir(os.path.join("/mount/resources/speech/corpora/CSS10", language, book)):
            for audio in os.listdir(os.path.join("/mount/resources/speech/corpora/CSS10", language, book)):
                if audio.endswith(".wav"):
                    file_list.append(os.path.join("/mount/resources/speech/corpora/CSS10", language, book, audio))
    return file_list


def get_file_list_css10ru():
    file_list = list()
    language = "russian"
    for book in os.listdir(os.path.join("/mount/resources/speech/corpora/CSS10", language)):
        if os.path.isdir(os.path.join("/mount/resources/speech/corpora/CSS10", language, book)):
            for audio in os.listdir(os.path.join("/mount/resources/speech/corpora/CSS10", language, book)):
                if audio.endswith(".wav"):
                    file_list.append(os.path.join("/mount/resources/speech/corpora/CSS10", language, book, audio))
    return file_list


def get_file_list_css10hu():
    file_list = list()
    language = "hungarian"
    for book in os.listdir(os.path.join("/mount/resources/speech/corpora/CSS10", language)):
        if os.path.isdir(os.path.join("/mount/resources/speech/corpora/CSS10", language, book)):
            for audio in os.listdir(os.path.join("/mount/resources/speech/corpora/CSS10", language, book)):
                if audio.endswith(".wav"):
                    file_list.append(os.path.join("/mount/resources/speech/corpora/CSS10", language, book, audio))
    return file_list


def get_file_list_css10du():
    file_list = list()
    language = "dutch"
    for book in os.listdir(os.path.join("/mount/resources/speech/corpora/CSS10", language)):
        if os.path.isdir(os.path.join("/mount/resources/speech/corpora/CSS10", language, book)):
            for audio in os.listdir(os.path.join("/mount/resources/speech/corpora/CSS10", language, book)):
                if audio.endswith(".wav"):
                    file_list.append(os.path.join("/mount/resources/speech/corpora/CSS10", language, book, audio))
    return file_list


def get_file_list_css10jp():
    file_list = list()
    language = "japanese"
    for book in os.listdir(os.path.join("/mount/resources/speech/corpora/CSS10", language)):
        if os.path.isdir(os.path.join("/mount/resources/speech/corpora/CSS10", language, book)):
            for audio in os.listdir(os.path.join("/mount/resources/speech/corpora/CSS10", language, book)):
                if audio.endswith(".wav"):
                    file_list.append(os.path.join("/mount/resources/speech/corpora/CSS10", language, book, audio))
    return file_list


def get_file_list_css10ch():
    file_list = list()
    language = "chinese"
    for book in os.listdir(os.path.join("/mount/resources/speech/corpora/CSS10", language)):
        if os.path.isdir(os.path.join("/mount/resources/speech/corpora/CSS10", language, book)):
            for audio in os.listdir(os.path.join("/mount/resources/speech/corpora/CSS10", language, book)):
                if audio.endswith(".wav"):
                    file_list.append(os.path.join("/mount/resources/speech/corpora/CSS10", language, book, audio))
    return file_list


def get_file_list_css10fr():
    file_list = list()
    language = "french"
    for book in os.listdir(os.path.join("/mount/resources/speech/corpora/CSS10", language)):
        if os.path.isdir(os.path.join("/mount/resources/speech/corpora/CSS10", language, book)):
            for audio in os.listdir(os.path.join("/mount/resources/speech/corpora/CSS10", language, book)):
                if audio.endswith(".wav"):
                    file_list.append(os.path.join("/mount/resources/speech/corpora/CSS10", language, book, audio))
    return file_list


def get_file_list_thorsten():
    file_list = list()
    with open("/mount/resources/speech/corpora/Thorsten_DE/metadata_shuf.csv", encoding="utf8") as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            file_list.append("/mount/resources/speech/corpora/Thorsten_DE/wavs/" + line.split("|")[0] + ".wav")
    return file_list


def get_file_list_nvidia_hifitts():
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

    return list(path_to_transcript.keys())


def get_file_list_spanish_blizzard_train():
    return list(build_path_to_transcript_dict_spanish_blizzard_train().keys())
