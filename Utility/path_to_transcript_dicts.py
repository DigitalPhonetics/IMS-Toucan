import os


def build_path_to_transcript_dict_hokuspokus():
    path_to_transcript = dict()
    for transcript_file in os.listdir("/mount/resources/speech/corpora/LibriVox.Hokuspokus/txt"):
        if transcript_file.endswith(".txt"):
            with open("/mount/resources/speech/corpora/LibriVox.Hokuspokus/txt/" + transcript_file, 'r',
                      encoding='utf8') as tf:
                transcript = tf.read()
            wav_path = "/mount/resources/speech/corpora/LibriVox.Hokuspokus/wav/" + transcript_file.rstrip(
                ".txt") + ".wav"
            path_to_transcript[wav_path] = transcript
    return path_to_transcript


def build_path_to_transcript_dict_libritts():
    path_train = "/mount/resources/speech/corpora/LibriTTS/train-clean-100"
    path_valid = "/mount/resources/speech/corpora/LibriTTS/dev-clean"

    path_to_transcript = dict()
    # we split training and validation differently, so we merge both folders into a single dict
    for speaker in os.listdir(path_train):
        for chapter in os.listdir(os.path.join(path_train, speaker)):
            for file in os.listdir(os.path.join(path_train, speaker, chapter)):
                if file.endswith("normalized.txt"):
                    with open(os.path.join(path_train, speaker, chapter, file), 'r',
                              encoding='utf8') as tf:
                        transcript = tf.read()
                    wav_file = file.split(".")[0] + ".wav"
                    path_to_transcript[os.path.join(path_train, speaker, chapter, wav_file)] = transcript
    for speaker in os.listdir(path_valid):
        for chapter in os.listdir(os.path.join(path_valid, speaker)):
            for file in os.listdir(os.path.join(path_valid, speaker, chapter)):
                if file.endswith("normalized.txt"):
                    with open(os.path.join(path_valid, speaker, chapter, file), 'r',
                              encoding='utf8') as tf:
                        transcript = tf.read()
                    wav_file = file.split(".")[0] + ".wav"
                    path_to_transcript[os.path.join(path_valid, speaker, chapter, wav_file)] = transcript
    return path_to_transcript


def build_path_to_transcript_dict_ljspeech():
    path_to_transcript = dict()
    for transcript_file in os.listdir("/mount/resources/speech/corpora/LJSpeech/16kHz/txt"):
        with open("/mount/resources/speech/corpora/LJSpeech/16kHz/txt/" + transcript_file, 'r', encoding='utf8') as tf:
            transcript = tf.read()
        wav_path = "/mount/resources/speech/corpora/LJSpeech/16kHz/wav/" + transcript_file.rstrip(".txt") + ".wav"
        path_to_transcript[wav_path] = transcript
    return path_to_transcript
