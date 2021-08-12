import re
import sys

import panphon
import phonemizer
import torch
from cleantext import clean


class ArticulatoryTextFrontend:

    def __init__(self,
                 language,
                 silent=True,
                 inference=False):
        """
        Language specific setup
        """
        self.inference = inference
        self.feature_table = panphon.FeatureTable()

        if language == "en":
            self.clean_lang = "en"
            self.g2p_lang = "en-us"
            self.expand_abbreviations = english_text_expansion
            if not silent:
                print("Created an English Text-Frontend")

        elif language == "de":
            self.clean_lang = "de"
            self.g2p_lang = "de"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a German Text-Frontend")

        else:
            print("Language not supported yet")
            sys.exit()

    def string_to_tensor(self, text, view=False):
        """
        Fixes unicode errors, expands some abbreviations,
        turns graphemes into phonemes and then vectorizes
        the sequence as articulatory features
        """
        phones = self.get_phone_string(text=text, include_eos_symbol=False)
        phone_segments = phones.split("~")
        articulatory_features_full = []
        for phone_segment in phone_segments:
            articulatory_features_seg = self.feature_table.word_to_vector_list(phone_segment, numeric=True)
            articulatory_features_seg_sil_dim_added = []
            for articulatory_features in articulatory_features_seg:
                articulatory_features_seg_sil_dim_added.append(articulatory_features + [0])
            articulatory_features_full += articulatory_features_seg_sil_dim_added + \
                                          [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
        articulatory_features_tensor = torch.FloatTensor(articulatory_features_full)

        if view:
            print("Phonemes: \n{}\n".format(phones))
            print("Features: \n{}\n".format(articulatory_features_tensor))
            print(len(phones))
            print(articulatory_features_tensor.shape)
        return articulatory_features_tensor

    def get_phone_string(self, text, include_eos_symbol=True):
        # clean unicode errors, expand abbreviations, handle emojis etc.
        utt = clean(text, fix_unicode=True, to_ascii=False, lower=False, lang=self.clean_lang)
        self.expand_abbreviations(utt)
        # phonemize
        phones = phonemizer.phonemize(utt,
                                      language_switch='remove-flags',
                                      backend="espeak",
                                      language=self.g2p_lang,
                                      preserve_punctuation=True,
                                      strip=True,
                                      punctuation_marks=';:,.!?¡¿—…"«»“”~/',
                                      with_stress=True).replace(";", ",").replace("/", " ") \
            .replace(":", ",").replace('"', ",").replace("-", ",").replace("-", ",").replace("\n", " ") \
            .replace("\t", " ").replace("¡", "").replace("¿", "").replace(",", "~")
        phones = re.sub("~+", "~", phones)
        phones = re.sub(r"\s+", " ", phones)
        if include_eos_symbol:
            phones += "#"
        return phones


def english_text_expansion(text):
    """
    Apply as small part of the tacotron style text cleaning pipeline, suitable for e.g. LJSpeech.
    See https://github.com/keithito/tacotron/
    Careful: Only apply to english datasets. Different languages need different cleaners.
    """
    _abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in
                      [('Mrs.', 'misess'), ('Mr.', 'mister'), ('Dr.', 'doctor'), ('St.', 'saint'), ('Co.', 'company'), ('Jr.', 'junior'), ('Maj.', 'major'),
                       ('Gen.', 'general'), ('Drs.', 'doctors'), ('Rev.', 'reverend'), ('Lt.', 'lieutenant'), ('Hon.', 'honorable'), ('Sgt.', 'sergeant'),
                       ('Capt.', 'captain'), ('Esq.', 'esquire'), ('Ltd.', 'limited'), ('Col.', 'colonel'), ('Ft.', 'fort')]]
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


if __name__ == '__main__':
    # test an English utterance
    tfr_en = ArticulatoryTextFrontend(language="en")
    tfr_en.string_to_tensor("Hello, world!", view=True)
