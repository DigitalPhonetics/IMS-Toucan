import re
import sys

import phonemizer
import torch

from Preprocessing.articulatory_features import generate_feature_table, get_phone_to_id


class ArticulatoryCombinedTextFrontend:

    def __init__(self,
                 language,
                 use_word_boundaries=False,  # goes together well with 
                 # parallel models and a aligner. Doesn't go together 
                 # well with autoregressive models.
                 use_explicit_eos=True,
                 use_prosody=False,  # unfortunately the non-segmental
                 # nature of prosodic markers mixed with the sequential
                 # phonemes hurts the performance of end-to-end models a
                 # lot, even though one might think enriching the input
                 # with such information would help.
                 use_lexical_stress=False,
                 silent=True,
                 allow_unknown=False,
                 add_silence_to_end=True,
                 strip_silence=True):
        """
        Mostly preparing ID lookups
        """
        self.strip_silence = strip_silence
        self.use_word_boundaries = use_word_boundaries
        self.allow_unknown = allow_unknown
        self.use_explicit_eos = use_explicit_eos
        self.use_prosody = use_prosody
        self.use_stress = use_lexical_stress
        self.add_silence_to_end = add_silence_to_end

        if language == "en":
            self.g2p_lang = "en-us"
            self.expand_abbreviations = english_text_expansion
            if not silent:
                print("Created an English Text-Frontend")

        elif language == "de":
            self.g2p_lang = "de"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a German Text-Frontend")

        elif language == "el":
            self.g2p_lang = "el"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Greek Text-Frontend")

        elif language == "es":
            self.g2p_lang = "es"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Spanish Text-Frontend")

        elif language == "fi":
            self.g2p_lang = "fi"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Finnish Text-Frontend")

        elif language == "ru":
            self.g2p_lang = "ru"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Russian Text-Frontend")

        elif language == "hu":
            self.g2p_lang = "hu"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Hungarian Text-Frontend")

        elif language == "nl":
            self.g2p_lang = "nl"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Dutch Text-Frontend")

        elif language == "fr":
            self.g2p_lang = "fr-fr"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a French Text-Frontend")

        elif language == "it":
            self.g2p_lang = "it"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Italian Text-Frontend")

        elif language == "pt":
            self.g2p_lang = "pt"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Portuguese Text-Frontend")

        elif language == "pl":
            self.g2p_lang = "pl"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Polish Text-Frontend")

        # remember to also update get_language_id() when adding something here

        else:
            print("Language not supported yet")
            sys.exit()

        self.phone_to_vector = generate_feature_table()

        self.phone_to_id = get_phone_to_id()

        self.id_to_phone = {v: k for k, v in self.phone_to_id.items()}

    def string_to_tensor(self, text, view=False, device="cpu", handle_missing=True, input_phonemes=False):
        """
        Fixes unicode errors, expands some abbreviations,
        turns graphemes into phonemes and then vectorizes
        the sequence as articulatory features
        """
        if input_phonemes:
            phones = text
        else:
            phones = self.get_phone_string(text=text, include_eos_symbol=True)
        if view:
            print("Phonemes: \n{}\n".format(phones))
        phones_vector = list()
        # turn into numeric vectors
        for char in phones:
            if handle_missing:
                try:
                    phones_vector.append(self.phone_to_vector[char])
                except KeyError:
                    print("unknown phoneme: {}".format(char))
            else:
                phones_vector.append(self.phone_to_vector[char])  # leave error handling to elsewhere

        return torch.Tensor(phones_vector, device=device)

    def get_phone_string(self, text, include_eos_symbol=True):
        # expand abbreviations
        utt = self.expand_abbreviations(text)
        # phonemize
        phones = phonemizer.phonemize(utt,
                                      language_switch='remove-flags',
                                      backend="espeak",
                                      language=self.g2p_lang,
                                      preserve_punctuation=True,
                                      strip=True,
                                      punctuation_marks=';:,.!?¡¿—…"«»“”~/',
                                      with_stress=self.use_stress).replace(";", ",").replace("/", " ").replace("—", "") \
            .replace(":", ",").replace('"', ",").replace("-", ",").replace("...", ",").replace("-", ",").replace("\n", " ") \
            .replace("\t", " ").replace("¡", "").replace("¿", "").replace(",", "~").replace(" ̃", "").replace('̩', "").replace("̃", "").replace("̪", "")
        # less than 1 wide characters hidden here
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
            phones = re.sub(" ", "~", phones)
        if self.strip_silence:
            phones = phones.lstrip("~").rstrip("~")
        if self.add_silence_to_end:
            phones += "~"  # adding a silence in the end during add_silence_to_end produces more natural sounding prosody
        if include_eos_symbol:
            phones += "#"

        phones = "~" + phones
        phones = re.sub("~+", "~", phones)

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


def get_language_id(language):
    if language == "de":
        return torch.LongTensor([1])
    elif language == "el":
        return torch.LongTensor([2])
    elif language == "es":
        return torch.LongTensor([3])
    elif language == "fi":
        return torch.LongTensor([4])
    elif language == "ru":
        return torch.LongTensor([5])
    elif language == "hu":
        return torch.LongTensor([6])
    elif language == "nl":
        return torch.LongTensor([7])
    elif language == "fr":
        return torch.LongTensor([8])
    elif language == "pt":
        return torch.LongTensor([9])
    elif language == "pl":
        return torch.LongTensor([10])
    elif language == "it":
        return torch.LongTensor([11])
    elif language == "en":
        return torch.LongTensor([12])


if __name__ == '__main__':
    # test an English utterance
    tf = ArticulatoryCombinedTextFrontend(language="en")
    print(tf.string_to_tensor("This is a complex sentence, it even has a pause! But can it do this? Nice.", view=True))

    tf = ArticulatoryCombinedTextFrontend(language="de")
    print(tf.string_to_tensor("Alles klar, jetzt testen wir einen deutschen Satz. Ich hoffe es gibt nicht mehr viele unspezifizierte Phoneme.", view=True))
