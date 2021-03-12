import os


def get_file_list_css10de():
    file_list = list()
    with open("Corpora/CSS10_DE/transcript.txt", encoding="utf8") as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            file_list.append("Corpora/CSS10_DE/" + line.split("|")[0])
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
