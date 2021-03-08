import sys
from collections import defaultdict

import numpy
import phonemizer
import torch
from cleantext import clean

"""
Explanation of the Tensor dimensions:

The First Block comes from a modified PanPhon phoneme 
vector lookup table and can optionally be used instead 
of a bland phoneme ID. It contains articulatory features 
of the phonemes. https://www.aclweb.org/anthology/C16-1328/

syl -- ternery, phonetic property of phoneme
son -- ternery, phonetic property of phoneme
cons -- ternery, phonetic property of phoneme
cont -- ternery, phonetic property of phoneme
delrel -- ternery, phonetic property of phoneme
lat -- ternery, phonetic property of phoneme
nas -- ternery, phonetic property of phoneme
strid -- ternery, phonetic property of phoneme
voi -- ternery, phonetic property of phoneme
sg -- ternery, phonetic property of phoneme
cg -- ternery, phonetic property of phoneme
ant -- ternery, phonetic property of phoneme
cor -- ternery, phonetic property of phoneme
distr -- ternery, phonetic property of phoneme
lab -- ternery, phonetic property of phoneme
hi -- ternery, phonetic property of phoneme
lo -- ternery, phonetic property of phoneme
back -- ternery, phonetic property of phoneme
round -- ternery, phonetic property of phoneme
velaric -- ternery, phonetic property of phoneme
tense -- ternery, phonetic property of phoneme
long -- ternery, phonetic property of phoneme
hitone -- ternery, phonetic property of phoneme
hireg -- ternery, phonetic property of phoneme

prosody and pauses -- integer, 1 = primary stress, 
                               2 = secondary stress,
                               3 = lengthening,
                               4 = half-lenghtening,
                               5 = shortening,
                               6 = syllable boundary,
                               7 = tact boundary,
                               8 = upper intonation grouping
                               9 = one of , ; : - identity
                               10 = intonation phrase boundary identity according to chinks and chunks
                             
sentence type -- integer, 1 = neutral,
                          2 = question,
                          3 = exclamation

"""


class TextFrontend:
    def __init__(self,
                 language,
                 use_panphon_vectors=True,
                 use_sentence_type=True,
                 use_word_boundaries=False,
                 use_explicit_eos=True
                 ):
        """
        Mostly preparing ID lookups
        """
        self.use_panphon_vectors = use_panphon_vectors
        self.use_sentence_type = use_sentence_type
        self.use_word_boundaries = use_word_boundaries
        self.use_explicit_eos = use_explicit_eos

        # list taken and modified from https://github.com/dmort27/panphon
        self.ipa_to_vector = defaultdict()
        if use_panphon_vectors:
            self.default_vector = [131, 131, 131, 131, 131, 131, 131, 131, 131, 131,
                                   131, 131, 131, 131, 131, 131, 131, 131, 131, 131,
                                   131, 131, 131, 131, 131]
        else:
            self.default_vector = 131
        with open("PreprocessingForTTS/ipa_vector_lookup.csv", encoding='utf8') as f:
            features = f.read()
        features_list = features.split("\n")
        for index in range(1, len(features_list)):
            line_list = features_list[index].split(",")
            if use_panphon_vectors:
                self.ipa_to_vector[line_list[0]] = [float(x) for x in line_list[1:]]
            else:
                self.ipa_to_vector[line_list[0]] = index

        if language == "en":
            self.clean_lang = "en"
            self.g2p_lang = "en-us"

        elif language == "de":
            self.clean_lang = "de"
            self.g2p_lang = "de"

        else:
            print("Language not supported yet")
            sys.exit()

    def string_to_tensor(self, text, view=False):
        """
        applies the entire pipeline to a text
        """
        # clean unicode errors etc
        utt = clean(text, fix_unicode=True, to_ascii=True, lower=False, lang=self.clean_lang)

        # phonemize
        phones = phonemizer.phonemize(utt,
                                      language_switch='remove-flags',
                                      backend="espeak",
                                      language=self.g2p_lang,
                                      preserve_punctuation=True,
                                      strip=True,
                                      with_stress=True).replace(",", "-").replace(";", "-").replace(":", "-")
        if view:
            print("Phonemes: \n{}\n".format(phones))

        tensors = []
        phones_vector = []

        # turn into numeric vectors
        for char in phones:
            if self.use_word_boundaries:
                if char != " ":
                    phones_vector.append(self.ipa_to_vector.get(char, self.default_vector))
            else:
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


if __name__ == '__main__':
    # test an English utterance
    tfr_en = TextFrontend(language="en",
                          use_panphon_vectors=False,
                          use_sentence_type=False,
                          use_word_boundaries=False,
                          use_explicit_eos=False)
    print(tfr_en.string_to_tensor("Hello!"))

    # test a German utterance
    tfr_de = TextFrontend(language="de",
                          use_panphon_vectors=True,
                          use_sentence_type=True,
                          use_word_boundaries=True,
                          use_explicit_eos=True)
    print(tfr_de.string_to_tensor("Hallo!"))

    tfr_autoreg = TextFrontend(language="en",
                               use_panphon_vectors=False,
                               use_sentence_type=False,
                               use_word_boundaries=False,
                               use_explicit_eos=True)
    print(tfr_autoreg.string_to_tensor("Hello there!"))
