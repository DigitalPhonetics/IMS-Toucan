import sys
from collections import defaultdict

import numpy
import phonemizer
import torch
from cleantext import clean


class TextFrontend:
    def __init__(self,
                 language,
                 use_panphon_vectors=True,
                 use_sentence_type=True,
                 use_word_boundaries=False,
                 use_explicit_eos=False):
        """
        Mostly preparing ID lookups
        """
        self.use_panphon_vectors = use_panphon_vectors
        self.use_sentence_type = use_sentence_type
        self.use_word_boundaries = use_word_boundaries
        self.use_explicit_eos = use_explicit_eos

        # list taken and modified from https://github.com/dmort27/panphon
        # see publication: https://www.aclweb.org/anthology/C16-1328/
        self.ipa_to_vector = defaultdict()
        if use_panphon_vectors:
            self.default_vector = [130, 130, 130, 130, 130, 130, 130, 130, 130, 130,
                                   130, 130, 130, 130, 130, 130, 130, 130, 130, 130,
                                   130, 130, 130, 130, 130]
        else:
            self.default_vector = 130
        with open("PreprocessingForTTS/ipa_vector_lookup.csv", encoding='utf8') as f:
            features = f.read()
        features_list = features.split("\n")
        for index in range(1, len(features_list)):
            line_list = features_list[index].split(",")
            if use_panphon_vectors:
                self.ipa_to_vector[line_list[0]] = [float(x) for x in line_list[1:]]
            else:
                self.ipa_to_vector[line_list[0]] = index
                # note: Index 0 is unused, so it can be used for padding as is convention.
                #       Index 1 is EOS, if you want to use explicit EOS.
                #       Index 130 is used for unknown characters
                #       Index 10 is used for pauses (heuristically)

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
                                      with_stress=False).replace(";", ",").replace(":", ",").replace('"', ",").replace(
            "--", ",").replace("\n", " ").replace("\t", " ").replace("  ", " ").replace("!", ".").replace("?", ".")
        if view:
            print("Phonemes: \n{}\n".format(phones))

        tensors = list()
        phones_vector = list()

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
