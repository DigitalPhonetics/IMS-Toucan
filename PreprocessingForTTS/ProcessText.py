import re
import sys
from collections import defaultdict

import numpy
import phonemizer
import torch
from cleantext import clean


class TextFrontend:
    def __init__(self,
                 language,
                 use_panphon_vectors=False,
                 use_word_boundaries=False,
                 use_explicit_eos=False,
                 use_prosody=False,  # unfortunately the non segmental
                 # nature of prosodic markers mixed with the sequential
                 # phonemes hurts the performance of end-to-end models a
                 # lot, even though one might think enriching the input
                 # with such information would help such systems.
                 use_lexical_stress=False,
                 path_to_panphon_table="PreprocessingForTTS/ipa_vector_lookup.csv",
                 silent=True):
        """
        Mostly preparing ID lookups
        """
        self.use_panphon_vectors = use_panphon_vectors
        self.use_word_boundaries = use_word_boundaries
        self.use_explicit_eos = use_explicit_eos
        self.use_prosody = use_prosody
        self.use_stress = use_lexical_stress

        # list taken and modified from https://github.com/dmort27/panphon
        # see publication: https://www.aclweb.org/anthology/C16-1328/
        self.ipa_to_vector = defaultdict()
        if use_panphon_vectors:
            self.default_vector = [132, 132, 132, 132, 132, 132, 132, 132, 132, 132,
                                   132, 132, 132, 132, 132, 132, 132, 132, 132, 132,
                                   132, 132, 132, 132, 132]
        else:
            self.default_vector = 132
        with open(path_to_panphon_table, encoding='utf8') as f:
            features = f.read()
        features_list = features.split("\n")
        for index in range(1, len(features_list)):
            line_list = features_list[index].split(",")
            if use_panphon_vectors:
                self.ipa_to_vector[line_list[0]] = [float(x) for x in line_list[1:]]
            else:
                self.ipa_to_vector[line_list[0]] = index
                # note: Index 0 is unused, so it can be used for padding as is convention.
                #       Index 1 is reserved for EOS, if you want to use explicit EOS.
                #       Index 132 is used for unknown characters
                #       Index 12 is used for pauses (heuristically)

        if language == "en":
            self.clean_lang = "en"
            self.g2p_lang = "en-us"
            self.expand_abbrevations = english_text_expansion
            if not silent:
                print("Created an English Text-Frontend")

        elif language == "de":
            self.clean_lang = "de"
            self.g2p_lang = "de"
            self.expand_abbrevations = lambda x: x
            if not silent:
                print("Created a German Text-Frontend")

        else:
            print("Language not supported yet")
            sys.exit()

    def string_to_tensor(self, text, view=False):
        """
        Fixes unicode errors, expands some abbreviations,
        turns graphemes into phonemes and then vectorizes
        the sequence either as IDs to be fed into an embedding
        layer, or as an articulatory matrix.
        """
        # clean unicode errors, expand abbreviations
        utt = clean(text, fix_unicode=True, to_ascii=False, lower=False, lang=self.clean_lang)
        self.expand_abbrevations(utt)

        # if an aligner has produced silence tokens before, turn
        # them into silence markers now so that they survive the
        # phonemizer:
        utt = utt.replace("_SIL_", "~")

        # phonemize
        phones = phonemizer.phonemize(utt,
                                      language_switch='remove-flags',
                                      backend="espeak",
                                      language=self.g2p_lang,
                                      preserve_punctuation=True,
                                      strip=True,
                                      punctuation_marks=';:,.!?¡¿—…"«»“”~',
                                      with_stress=self.use_stress).replace(";", ",").replace(":", ",").replace('"',
                                                                                                               ",").replace(
            "--", ",").replace("-", ",").replace("\n", " ").replace("\t", " ").replace("¡", "!").replace(
            "¿", "?").replace(",", "~").replace("~~", "~")

        if not self.use_prosody:
            # retain ~ as heuristic pause marker, even though all other symbols are removed with this option.
            # also retain . ? and ! since they can be indicators for the stop token
            phones = phones.replace("ˌ", "").replace("ː", "").replace(
                "ˑ", "").replace("˘", "").replace("|", "").replace("‖", "")

        if not self.use_word_boundaries:
            phones = phones.replace(" ", "")

        if view:
            print("Phonemes: \n{}\n".format(phones))

        tensors = list()
        phones_vector = list()

        # turn into numeric vectors
        for char in phones:
            phones_vector.append(self.ipa_to_vector.get(char, self.default_vector))

        if self.use_explicit_eos:
            phones_vector.append(self.ipa_to_vector["end_of_input"])

        # turn into tensors
        if self.use_panphon_vectors:
            for line in numpy.transpose(numpy.array(phones_vector)):
                tensors.append(torch.LongTensor(line))
        else:
            tensors.append(torch.LongTensor(phones_vector))

        # combine tensors and return
        return torch.stack(tensors, 0)

    def get_phone_string(self, text):
        utt = clean(text, fix_unicode=True, to_ascii=False, lower=False, lang=self.clean_lang)
        self.expand_abbrevations(utt)
        utt = utt.replace("_SIL_", "~")
        phones = phonemizer.phonemize(utt,
                                      language_switch='remove-flags',
                                      backend="espeak",
                                      language=self.g2p_lang,
                                      preserve_punctuation=True,
                                      strip=True,
                                      punctuation_marks=';:,.!?¡¿—…"«»“”~',
                                      with_stress=self.use_stress).replace(";", ",").replace(":", ",").replace('"',
                                                                                                               ",").replace(
            "--", ",").replace("-", ",").replace("\n", " ").replace("\t", " ").replace("¡", "!").replace(
            "¿", "?").replace(",", "~")
        if not self.use_prosody:
            phones = phones.replace("ˌ", "").replace("ː", "").replace(
                "ˑ", "").replace("˘", "").replace("|", "").replace("‖", "")
        if not self.use_word_boundaries:
            phones = phones.replace(" ", "")
        return phones + "#"


def english_text_expansion(text):
    """
    Apply as small part of the tacotron style text cleaning pipeline, suitable for e.g. LJSpeech.
    See https://github.com/keithito/tacotron/
    Careful: Only apply to english datasets. Different languages need different cleaners.
    """
    _abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
        ('Mrs.', 'misess'),
        ('Mr.', 'mister'),
        ('Dr.', 'doctor'),
        ('St.', 'saint'),
        ('Co.', 'company'),
        ('Jr.', 'junior'),
        ('Maj.', 'major'),
        ('Gen.', 'general'),
        ('Drs.', 'doctors'),
        ('Rev.', 'reverend'),
        ('Lt.', 'lieutenant'),
        ('Hon.', 'honorable'),
        ('Sgt.', 'sergeant'),
        ('Capt.', 'captain'),
        ('Esq.', 'esquire'),
        ('Ltd.', 'limited'),
        ('Col.', 'colonel'),
        ('Ft.', 'fort')]]
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


if __name__ == '__main__':
    # test an English utterance
    tfr_en = TextFrontend(language="en",
                          use_panphon_vectors=False,
                          use_word_boundaries=False,
                          use_explicit_eos=False,
                          path_to_panphon_table="ipa_vector_lookup.csv")
    print(tfr_en.string_to_tensor("Hello world, this is a test!", view=True))

    # test a German utterance
    tfr_de = TextFrontend(language="de",
                          use_panphon_vectors=False,
                          use_word_boundaries=False,
                          use_explicit_eos=False,
                          path_to_panphon_table="ipa_vector_lookup.csv")
    print(tfr_de.string_to_tensor("Hallo Welt, dies ist ein test!", view=True))
