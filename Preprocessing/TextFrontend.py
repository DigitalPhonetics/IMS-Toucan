import re
import sys
from collections import defaultdict

import phonemizer
import torch
from cleantext import clean


class TextFrontend:

    def __init__(self,
                 language,
                 use_word_boundaries=False,
                 use_explicit_eos=False,
                 use_prosody=False,  # unfortunately the non-segmental
                 # nature of prosodic markers mixed with the sequential
                 # phonemes hurts the performance of end-to-end models a
                 # lot, even though one might think enriching the input
                 # with such information would help such systems.
                 use_lexical_stress=False,
                 path_to_phoneme_list="Preprocessing/ipa_list.txt",
                 silent=True,
                 allow_unknown=False,
                 inference=False):
        """
        Mostly preparing ID lookups
        """
        self.use_word_boundaries = use_word_boundaries
        self.allow_unknown = allow_unknown
        self.use_explicit_eos = use_explicit_eos
        self.use_prosody = use_prosody
        self.use_stress = use_lexical_stress
        self.inference = inference
        if allow_unknown:
            self.ipa_to_vector = defaultdict()
            self.default_vector = 165
        else:
            self.ipa_to_vector = dict()
        with open(path_to_phoneme_list, "r", encoding='utf8') as f:
            phonemes = f.read()
            # using https://github.com/espeak-ng/espeak-ng/blob/master/docs/phonemes.md
        phoneme_list = phonemes.split("\n")
        for index in range(1, len(phoneme_list)):
            self.ipa_to_vector[phoneme_list[index]] = index
            # note: Index 0 is unused, so it can be used for padding as is convention.
            #       Index 1 is reserved for end_of_utterance
            #       Index 2 is reserved for begin of sentence token
            #       Index 13 is used for pauses (heuristically)

        # The point of having the phonemes in a separate file is to ensure reproducibility.
        # The line of the phoneme is the ID of the phoneme, so you can have multiple such
        # files and always just supply the one during inference which you used during training.

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
        the sequence as IDs to be fed into an embedding
        layer
        """
        phones = self.get_phone_string(text=text, include_eos_symbol=False)
        if self.inference:
            phones += "~"  # adding a silence in the end during inference produces more natural sounding prosody
        if view:
            print("Phonemes: \n{}\n".format(phones))
        phones_vector = list()
        # turn into numeric vectors
        for char in phones:
            if self.allow_unknown:
                phones_vector.append(self.ipa_to_vector.get(char, self.default_vector))
            else:
                try:
                    phones_vector.append(self.ipa_to_vector[char])
                except KeyError:
                    print("unknown phoneme: {}".format(char))
        if self.use_explicit_eos:
            phones_vector.append(self.ipa_to_vector["end_of_input"])
        return torch.LongTensor(phones_vector).unsqueeze(0)

    def get_phone_string(self, text, include_eos_symbol=True):
        # clean unicode errors, expand abbreviations, handle emojis etc.
        utt = clean(text, fix_unicode=True, to_ascii=False, lower=False, lang=self.clean_lang)
        self.expand_abbreviations(utt)
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
                                      punctuation_marks=';:,.!?¡¿—…"«»“”~/',
                                      with_stress=self.use_stress).replace(";", ",").replace("/", " ") \
            .replace(":", ",").replace('"', ",").replace("-", ",").replace("-", ",").replace("\n", " ") \
            .replace("\t", " ").replace("¡", "").replace("¿", "").replace(",", "~")
        phones = re.sub("~+", "~", phones)
        if not self.use_prosody:
            # retain ~ as heuristic pause marker, even though all other symbols are removed with this option.
            # also retain . ? and ! since they can be indicators for the stop token
            phones = phones.replace("ˌ", "").replace("ː", "").replace("ˑ", "") \
                .replace("˘", "").replace("|", "").replace("‖", "")
        if not self.use_word_boundaries:
            phones = phones.replace(" ", "")
        else:
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
    tfr_en = TextFrontend(language="en", use_word_boundaries=False, use_explicit_eos=False,
                          path_to_phoneme_list="ipa_list.txt")
    print(tfr_en.string_to_tensor("Hello world, this is a test!", view=True))

    # test a German utterance
    tfr_de = TextFrontend(language="de", use_word_boundaries=False, use_explicit_eos=False,
                          path_to_phoneme_list="ipa_list.txt")
    print(tfr_de.string_to_tensor("Hallo Welt, dies ist ein Test!", view=True))
