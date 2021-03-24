from InferenceInterfaces.SingleSpeakerTransformerTTSInference import SingleSpeakerTransformerTTSInference


def read_texts(lang, sentence, filename):
    tts = SingleSpeakerTransformerTTSInference(lang=lang)
    if type(sentence) == str:
        sentence = [sentence]
    tts.read_to_file(text_list=sentence, file_location=filename)


if __name__ == '__main__':
    read_texts(lang="en",
               sentence=["Hello world.",
                         "This is a bunch of sentences."],
               filename="test_en.wav")
    read_texts(lang="de",
               sentence=["Hallo Welt.",
                         "Dies hier sind ein paar Sätze."],
               filename="test_de.wav")
