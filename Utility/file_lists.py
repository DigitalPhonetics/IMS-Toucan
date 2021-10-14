import os


def get_file_list_karlsson():
    root = "/mount/resources/speech/corpora/MAILabs_german_single_speaker_karlsson"
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


def get_file_list_eva():
    root = "/mount/resources/speech/corpora/MAILabs_german_single_speaker_eva"
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


def get_file_list_hokuspokus():
    file_list = list()
    for wav_file in os.listdir("/mount/resources/speech/corpora/LibriVox.Hokuspokus/wav"):
        if ".wav" in wav_file:
            file_list.append("/mount/resources/speech/corpora/LibriVox.Hokuspokus/wav/" + wav_file)
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
