from InferenceInterfaces.SingleSpeakerTransformerTTSInference import SingleSpeakerTransformerTTSInference


def read_texts(lang, sentence, filename, reduction_factor=1):
    tts = SingleSpeakerTransformerTTSInference(lang=lang, reduction_factor=reduction_factor)
    if type(sentence) == str:
        sentence = [sentence]
    tts.read_to_file(text_list=sentence, file_location=filename)


if __name__ == '__main__':
    read_texts(lang="en",
               sentence=["This is how the Transformer Synthesis sounds after just one hour of training.",
                         "Unfortunately this only works with a reduction factor of 5 at the moment."],
               filename="test_en.wav",
               reduction_factor=1)

    read_texts(lang="de",
               sentence=["Hallo Welt.",
                         "Dies hier sind ein paar Sätze."],
               filename="test_de.wav",
               reduction_factor=1)
