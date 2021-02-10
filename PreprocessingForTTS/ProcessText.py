import sys
from collections import defaultdict

import numpy
import phonemizer
import spacy
import torch
from cleantext import clean
from spacy.cli import download
from spacy.lang.de import German
from spacy.lang.en import English

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

The POS feature is assumed to help with the latent learning of 
intonation phrase boundaries, as the Chinks and Chunks theory 
suggests, see https://www.researchgate.net/publication/230876257_Text_Analysis_and_Word_Pronunciation_in_Text-to-Speech_Synthesis

pos -- integer, 1 = content word, 
                2 = function word, 
                3 = other word, 

The position feature is not necessary in transformers due to 
the positional encoding, but likely helpful otherwise.

position in sequence -- float, corresponds to percent of sequence from left to right. 
"""


class TextFrontend:
    def __init__(self,
                 language,
                 use_panphon_vectors=True,
                 use_shallow_pos=False,
                 use_chinksandchunks_ipb=True,
                 use_positional_information=False,
                 use_word_boundaries=False,
                 use_sentence_type=True,
                 use_explicit_eos=True
                 ):
        """
        Mostly loading the right spacy
        models and preparing ID lookups
        """
        self.use_panphon_vectors = use_panphon_vectors
        self.use_shallow_pos = use_shallow_pos
        self.use_chinksandchunks_ipb = use_chinksandchunks_ipb
        self.use_positional_information = use_positional_information
        self.use_word_boundaries = use_word_boundaries
        self.use_sentence_type = use_sentence_type
        self.use_explicit_eos = use_explicit_eos

        # list taken and modified from https://github.com/dmort27/panphon
        self.ipa_to_vector = defaultdict()
        if use_panphon_vectors:
            self.default_vector = [131, 131, 131, 131, 131, 131, 131, 131, 131, 131,
                                   131, 131, 131, 131, 131, 131, 131, 131, 131, 131,
                                   131, 131, 131, 131, 131]
        else:
            self.default_vector = 131
        with open("ipa_vector_lookup.csv", encoding='utf8') as f:
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
            if use_chinksandchunks_ipb or use_shallow_pos:
                try:
                    self.nlp = spacy.load('en_core_web_sm')
                except IOError:
                    download("en_core_web_sm")
                    self.nlp = spacy.load('en_core_web_sm')
            else:
                self.nlp = English()

        elif language == "de":
            self.clean_lang = "de"
            self.g2p_lang = "de"
            if use_chinksandchunks_ipb or use_shallow_pos:
                try:
                    self.nlp = spacy.load('de_core_news_sm')
                except IOError:
                    download("de_core_news_sm")
                    self.nlp = spacy.load('de_core_news_sm')
            else:
                self.nlp = German()

        else:
            print("Language not supported yet")
            sys.exit()

        if self.use_shallow_pos:
            content_word_tags = {"ADJ", "ADV", "INTJ", "NOUN", "PROPN", "VERB"}
            function_word_tags = {"ADP", "AUX", "CCONJ", "DET", "NUM", "PART", "PRON", "SCONJ"}
            other_tags = {"PUNCT", "SYM", "X"}
            self.tag_id_lookup = {"SPACE__": 3}
            for tag in content_word_tags:
                self.tag_id_lookup[tag] = 1
            for tag in function_word_tags:
                self.tag_id_lookup[tag] = 2
            for tag in other_tags:
                self.tag_id_lookup[tag] = 3

    def string_to_tensor(self, text, view=False):
        """
        applies the entire pipeline to a text
        """
        # tokenize
        utt = self.nlp(text)

        # clean unicode errors etc
        cleaned_tokens = []
        for index, token in enumerate(utt):
            cleaned_tokens.append(clean(token, fix_unicode=True, to_ascii=True, lower=False, lang=self.clean_lang))
        if view:
            print("Cleaned Tokens: \n{}\n".format(cleaned_tokens))

        # phonemize
        phonemized_tokens = []
        for cleaned_token in cleaned_tokens:
            phonemized_tokens.append(phonemizer.phonemize(cleaned_token,
                                                          backend="espeak",
                                                          language=self.g2p_lang,
                                                          preserve_punctuation=True,
                                                          strip=True,
                                                          with_stress=True).replace(",",
                                                                                    "-").replace(";",
                                                                                                 "-").replace(":",
                                                                                                              "-"))
        if view:
            print("Phonemes: \n{}\n".format(phonemized_tokens))

        tensors = []
        phones_vector = []
        tags_vector = []
        position_vector = []

        # vectorize and get POS
        sent_type_dim = 0
        for index, phonemized_token in enumerate(phonemized_tokens):
            for char in phonemized_token:
                if char == "!":
                    sent_type_dim = 1
                elif char == "?":
                    sent_type_dim = 2
                else:
                    phones_vector.append(self.ipa_to_vector.get(char, self.default_vector))
                    if self.use_shallow_pos or self.use_chinksandchunks_ipb:
                        tags_vector.append(utt[index].pos_)
                    if self.use_chinksandchunks_ipb:
                        if len(tags_vector) > 2:
                            if tags_vector[-2] != 2 and tags_vector[-1] == 2:
                                phones_vector.append(self.ipa_to_vector["intonation_phrase_boundary"])
                                if self.use_shallow_pos:
                                    tags_vector.append(utt[index].pos_)
            if self.use_word_boundaries:
                if index < len(phonemized_tokens) - 1:
                    phones_vector.append(self.default_vector)
                    if self.use_shallow_pos:
                        tags_vector.append("SPACE__")
            if self.use_explicit_eos:
                if not index < len(phonemized_tokens) - 1:
                    phones_vector.append(self.ipa_to_vector["end_of_input"])
                    if self.use_shallow_pos:
                        tags_vector.append("SPACE__")

        # generate tensors
        if not self.default_vector == 0:
            for line in numpy.transpose(numpy.array(phones_vector)):
                tensors.append(torch.tensor(line))
        else:
            tensors.append(torch.tensor(phones_vector))

        if self.use_sentence_type:
            tensors.append(torch.tensor([sent_type_dim] * tensors[-1].shape[0]))

        if self.use_shallow_pos:
            tags_numeric_vector = []
            for el in tags_vector:
                tags_numeric_vector.append(self.tag_id_lookup[el])
            tensors.append(torch.tensor(tags_numeric_vector))

        if self.use_positional_information:
            for index in range(len(phones_vector)):
                position_vector.append(round(index / len(phones_vector), 3))
            tensors.append(torch.tensor(position_vector))

        if view and self.use_shallow_pos:
            tags = []
            for el in utt:
                tags.append(el.tag_)
            print("POS Tags: \n{}\n".format(tags))

        # combine tensors and return
        return torch.stack(tensors, 0)


if __name__ == '__main__':
    # test an English utterance
    tfr_en = TextFrontend(language="en",
                          use_panphon_vectors=False,
                          use_shallow_pos=False,
                          use_sentence_type=False,
                          use_positional_information=False,
                          use_word_boundaries=False,
                          use_chinksandchunks_ipb=False,
                          use_explicit_eos=False)
    print(tfr_en.string_to_tensor("Hello!"))

    # test a German utterance
    tfr_de = TextFrontend(language="de",
                          use_panphon_vectors=True,
                          use_shallow_pos=True,
                          use_sentence_type=True,
                          use_positional_information=True,
                          use_word_boundaries=True,
                          use_chinksandchunks_ipb=True,
                          use_explicit_eos=True)
    print(tfr_de.string_to_tensor("Hallo!"))

    tfr_autoreg = TextFrontend(language="en",
                               use_panphon_vectors=False,
                               use_shallow_pos=False,
                               use_sentence_type=False,
                               use_positional_information=False,
                               use_word_boundaries=False,
                               use_chinksandchunks_ipb=True,
                               use_explicit_eos=True)
    print(tfr_autoreg.string_to_tensor("Hello there!"))
