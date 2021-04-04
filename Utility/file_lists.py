import os


def get_file_list_hokuspokus():
    file_list = list()
    for wav_file in os.listdir("/mount/resources/speech/corpora/LibriVox.Hokuspokus/wav"):
        if ".wav" in wav_file:
            file_list.append("/mount/resources/speech/corpora/LibriVox.Hokuspokus/wav/" + wav_file)
    return file_list


def get_file_list_libritts():
    path_train = "/mount/resources/speech/corpora/LibriTTS/train-clean-100"
    path_valid = "/mount/resources/speech/corpora/LibriTTS/dev-clean"
    file_list = list()
    # we split training and validation differently, so we merge both folders into a single dict
    for speaker in os.listdir(path_train):
        for chapter in os.listdir(os.path.join(path_train, speaker)):
            for file in os.listdir(os.path.join(path_train, speaker, chapter)):
                if file.endswith(".wav"):
                    file_list.append(os.path.join(path_train, speaker, chapter, file))
    for speaker in os.listdir(path_valid):
        for chapter in os.listdir(os.path.join(path_valid, speaker)):
            for file in os.listdir(os.path.join(path_valid, speaker, chapter)):
                if file.endswith("normalized.txt"):
                    if file.endswith(".wav"):
                        file_list.append(os.path.join(path_train, speaker, chapter, file))
    return file_list


def get_file_list_ljspeech():
    file_list = list()
    for wav_file in os.listdir("/mount/resources/speech/corpora/LJSpeech/16kHz/wav"):
        if ".wav" in wav_file:
            file_list.append("/mount/resources/speech/corpora/LJSpeech/16kHz/wav/" + wav_file)
    return file_list


def get_file_list_css10ge():
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
