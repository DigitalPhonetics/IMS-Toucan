import re
import sys

import panphon
import phonemizer
import torch
from cleantext import clean


class ArticulatoryPanphonTextFrontend:

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

        elif language == "el":
            self.clean_lang = None
            self.g2p_lang = "el"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Greek Text-Frontend")

        elif language == "es":
            self.clean_lang = None
            self.g2p_lang = "es"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Spanish Text-Frontend")

        elif language == "fi":
            self.clean_lang = None
            self.g2p_lang = "fi"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Finnish Text-Frontend")

        elif language == "ru":
            self.clean_lang = None
            self.g2p_lang = "ru"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Russian Text-Frontend")

        elif language == "hu":
            self.clean_lang = None
            self.g2p_lang = "hu"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Hungarian Text-Frontend")

        elif language == "nl":
            self.clean_lang = None
            self.g2p_lang = "nl"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Dutch Text-Frontend")

        elif language == "fr":
            self.clean_lang = None
            self.g2p_lang = "fr"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a French Text-Frontend")

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
        phone_segments_between_pauses = phones.split("~")
        articulatory_features_full_utt = []
        for phone_segment_between_pauses in phone_segments_between_pauses:
            articulatory_features_between_pauses = []
            for word in phone_segment_between_pauses.split():
                articulatory_features_seg = self.feature_table.word_to_vector_list(word, numeric=True)
                articulatory_features_seg_sil_dim_added = []
                for articulatory_features in articulatory_features_seg:
                    articulatory_features_seg_sil_dim_added.append(articulatory_features + [0])
                articulatory_features_between_pauses += articulatory_features_seg_sil_dim_added
                if "?" in word:
                    articulatory_features_between_pauses += [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
            articulatory_features_full_utt += articulatory_features_between_pauses + [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]]  # silence
        # now cut out the silence that is added when splitting, then add end of sentence
        articulatory_features_full_utt = articulatory_features_full_utt[:-1] + [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        articulatory_features_tensor = torch.FloatTensor(articulatory_features_full_utt)

        if view:
            print("Phonemes: \n{}\n".format(self.get_phone_string(text=text, include_eos_symbol=True, for_labelling=True)))
            print("Features: \n{}\n".format(articulatory_features_tensor))
            print(articulatory_features_tensor.shape)
        return articulatory_features_tensor

    def get_phone_string(self, text, include_eos_symbol=True, for_labelling=False):
        # clean unicode errors, expand abbreviations, handle emojis etc.
        if self.clean_lang is not None:
            utt = clean(text, fix_unicode=True, to_ascii=False, lower=False, lang=self.clean_lang)
        else:
            utt = text
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
            .replace("\t", " ").replace("¡", "").replace("¿", "").replace(",", "~").replace(" ̃", "").replace('̩', "")
        phones = re.sub("~+", "~", phones)
        phones = re.sub(r"\s+", " ", phones)
        if include_eos_symbol:
            phones += "#"
        if for_labelling:
            phone_segments_between_pauses = phones.split("~")
            ipa_segments_full_utt = []
            for phone_segment_between_pauses in phone_segments_between_pauses:
                for word in phone_segment_between_pauses.split():
                    ipa_segment_of_word = self.feature_table.ipa_segs(word)
                    ipa_segments_full_utt += ipa_segment_of_word
                    if "?" in word:
                        ipa_segments_full_utt += "?"
                ipa_segments_full_utt += ['~']
            ipa_segments_full_utt = ipa_segments_full_utt[:-1] + ['#']
            return ipa_segments_full_utt
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
    tfr_en = ArticulatoryPanphonTextFrontend(language="en")
    tfr_en.string_to_tensor("This is a complex sentence, it even has a pause! But can it do this?", view=True)
