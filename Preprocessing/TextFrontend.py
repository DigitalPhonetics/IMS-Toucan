# -*- coding: utf-8 -*-

import json
import re
import sys
from pathlib import Path

import torch
from dragonmapper.transcriptions import pinyin_to_ipa
from phonemizer.backend import EspeakBackend
from pypinyin import pinyin

from Preprocessing.articulatory_features import generate_feature_table
from Preprocessing.articulatory_features import get_feature_to_index_lookup
from Preprocessing.articulatory_features import get_phone_to_id
from Preprocessing.french_pos import map_to_wiktionary_pos
from Utility.storage_config import PREPROCESSING_DIR


class ArticulatoryCombinedTextFrontend:

    def __init__(self,
                 language,
                 use_explicit_eos=True,
                 use_lexical_stress=True,
                 silent=True,
                 add_silence_to_end=True,
                 use_word_boundaries=True):
        """
        Mostly preparing ID lookups
        """
        self.language = language
        self.use_explicit_eos = use_explicit_eos
        self.use_stress = use_lexical_stress
        self.add_silence_to_end = add_silence_to_end
        self.use_word_boundaries = use_word_boundaries

        register_to_height = {
            "˥": 5,
            "˦": 4,
            "˧": 3,
            "˨": 2,
            "˩": 1
        }
        self.rising_perms = list()
        self.falling_perms = list()
        self.peaking_perms = list()
        self.dipping_perms = list()

        for first_tone in ["˥", "˦", "˧", "˨", "˩"]:
            for second_tone in ["˥", "˦", "˧", "˨", "˩"]:
                if register_to_height[first_tone] > register_to_height[second_tone]:
                    self.falling_perms.append(first_tone + second_tone)
                else:
                    self.rising_perms.append(first_tone + second_tone)
                for third_tone in ["˥", "˦", "˧", "˨", "˩"]:
                    if register_to_height[first_tone] > register_to_height[second_tone] < register_to_height[third_tone]:
                        self.dipping_perms.append(first_tone + second_tone + third_tone)
                    elif register_to_height[first_tone] < register_to_height[second_tone] > register_to_height[third_tone]:
                        self.peaking_perms.append(first_tone + second_tone + third_tone)

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
            from flair.models import SequenceTagger
            import flair
            # have to import down here, because this import sets the cuda visible devices GLOBALLY for some reason.
            self.g2p_lang = "fr-fr"
            self.expand_abbreviations = lambda x: x
            # add POS Tagger for Blizzard Challenge
            flair.cache_root = Path(f"{PREPROCESSING_DIR}/.flair")
            self.pos_tagger = SequenceTagger.load("qanastek/pos-french-camembert-flair")
            self.homographs = load_json_from_path("Preprocessing/french_homographs_preprocessed.json")
            self.homograph_list = list(self.homographs.keys())
            self.poet_to_wiktionary = map_to_wiktionary_pos()

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

        elif language == "pt-br":
            self.g2p_lang = "pt-br"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Brazilian Portuguese Text-Frontend")

        elif language == "pl":
            self.g2p_lang = "pl"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Polish Text-Frontend")

        elif language == "cmn":
            self.g2p_lang = "cmn"  # we don't use espeak for this case
            self.expand_abbreviations = convert_kanji_to_pinyin_mandarin
            if not silent:
                print("Created a Mandarin-Chinese Text-Frontend")

        elif language == "vi":
            self.g2p_lang = "vi"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Northern-Vietnamese Text-Frontend")

        elif language == "uk":
            self.g2p_lang = "uk"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Ukrainian Text-Frontend")

        elif language == "fa":
            self.g2p_lang = "fa"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Farsi Text-Frontend")

        else:
            print("Language not supported yet")
            sys.exit()

        # remember to also update get_language_id() below when adding something here, as well as the get_example_sentence function

        if self.g2p_lang != "cmn" and self.g2p_lang != "cmn-latn-pinyin":
            self.phonemizer_backend = EspeakBackend(language=self.g2p_lang,
                                                    punctuation_marks=';:,.!?¡¿—…"«»“”~/。【】、‥،؟“”؛',
                                                    preserve_punctuation=True,
                                                    language_switch='remove-flags',
                                                    with_stress=self.use_stress)

        self.phone_to_vector = generate_feature_table()
        self.phone_to_id = get_phone_to_id()
        self.id_to_phone = {v: k for k, v in self.phone_to_id.items()}

    @staticmethod
    def get_example_sentence(lang):
        if lang == "en":
            return "This is a complex sentence, it even has a pause!"
        elif lang == "de":
            return "Dies ist ein komplexer Satz, er hat sogar eine Pause!"
        elif lang == "el":
            return "Αυτή είναι μια σύνθετη πρόταση, έχει ακόμη και παύση!"
        elif lang == "es":
            return "Esta es una oración compleja, ¡incluso tiene una pausa!"
        elif lang == "fi":
            return "Tämä on monimutkainen lause, sillä on jopa tauko!"
        elif lang == "ru":
            return "Это сложное предложение, в нем даже есть пауза!"
        elif lang == "hu":
            return "Ez egy összetett mondat, még szünet is van benne!"
        elif lang == "nl":
            return "Dit is een complexe zin, er zit zelfs een pauze in!"
        elif lang == "fr":
            return "C'est une phrase complexe, elle a même une pause !"
        elif lang == "pt" or lang == "pt-br":
            return "Esta é uma frase complexa, tem até uma pausa!"
        elif lang == "pl":
            return "To jest zdanie złożone, ma nawet pauzę!"
        elif lang == "it":
            return "Questa è una frase complessa, ha anche una pausa!"
        elif lang == "cmn":
            return "这是一个复杂的句子，它甚至包含一个停顿。"
        elif lang == "vi":
            return "Đây là một câu phức tạp, nó thậm chí còn chứa một khoảng dừng."
        else:
            print(f"No example sentence specified for the language: {lang}\n "
                  f"Please specify an example sentence in the get_example_sentence function in Preprocessing/TextFrontend to track your progress.")
            return None

    def string_to_tensor(self, text, view=False, device="cpu", handle_missing=True, input_phonemes=False):
        """
        Fixes unicode errors, expands some abbreviations,
        turns graphemes into phonemes and then vectorizes
        the sequence as articulatory features
        """
        if input_phonemes:
            phones = text
        else:
            phones = self.get_phone_string(text=text, include_eos_symbol=True, for_feature_extraction=True)
        phones = phones.replace("ɚ", "ə").replace("ᵻ", "ɨ")
        if view:
            print("Phonemes: \n{}\n".format(phones))
        phones_vector = list()
        # turn into numeric vectors
        stressed_flag = False

        for char in phones:
            # affects following phoneme -----------------
            if char == '\u02C8':
                # primary stress
                stressed_flag = True
            # affects previous phoneme -----------------
            elif char == '\u02D0':
                # lengthened
                phones_vector[-1][get_feature_to_index_lookup()["lengthened"]] = 1
            elif char == '\u02D1':
                # half length
                phones_vector[-1][get_feature_to_index_lookup()["half-length"]] = 1
            elif char == '\u0306':
                # shortened
                phones_vector[-1][get_feature_to_index_lookup()["shortened"]] = 1
            elif char == '̃':
                # nasalized (vowel)
                phones_vector[-1][get_feature_to_index_lookup()["nasal"]] = 1
            elif char == "˥":
                # very high tone
                phones_vector[-1][get_feature_to_index_lookup()["very-high-tone"]] = 1
            elif char == "˦":
                # high tone
                phones_vector[-1][get_feature_to_index_lookup()["high-tone"]] = 1
            elif char == "˧":
                # mid tone
                phones_vector[-1][get_feature_to_index_lookup()["mid-tone"]] = 1
            elif char == "˨":
                # low tone
                phones_vector[-1][get_feature_to_index_lookup()["low-tone"]] = 1
            elif char == "˩":
                # very low tone
                phones_vector[-1][get_feature_to_index_lookup()["very-low-tone"]] = 1
            elif char == "⭧":
                # rising tone
                phones_vector[-1][get_feature_to_index_lookup()["rising-tone"]] = 1
            elif char == "⭨":
                # falling tone
                phones_vector[-1][get_feature_to_index_lookup()["falling-tone"]] = 1
            elif char == "⮁":
                # peaking tone
                phones_vector[-1][get_feature_to_index_lookup()["peaking-tone"]] = 1
            elif char == "⮃":
                # dipping tone
                phones_vector[-1][get_feature_to_index_lookup()["dipping-tone"]] = 1
            else:
                if handle_missing:
                    try:
                        phones_vector.append(self.phone_to_vector[char].copy())
                    except KeyError:
                        print("unknown phoneme: {}".format(char))
                else:
                    phones_vector.append(self.phone_to_vector[char].copy())  # leave error handling to elsewhere

                if stressed_flag:
                    stressed_flag = False
                    phones_vector[-1][get_feature_to_index_lookup()["stressed"]] = 1

        return torch.Tensor(phones_vector, device=device)

    def get_phone_string(self, text, include_eos_symbol=True, for_feature_extraction=False, for_plot_labels=False, resolve_homographs=True):
        # expand abbreviations
        utt = self.expand_abbreviations(text)

        # phonemize
        if self.g2p_lang == "cmn-latn-pinyin" or self.g2p_lang == "cmn":
            phones = pinyin_to_ipa(utt)
        elif self.g2p_lang == "fr-fr" and resolve_homographs:
            from flair.data import Sentence
            sentence = Sentence(utt)
            self.pos_tagger.predict(sentence)
            # print(sentence.to_tagged_string())

            phones = ''  # we'll bulid the phone string incrementally
            chunk_to_phonemize = ''
            labels = sentence.get_labels()
            for i,label in enumerate(labels):
                token = label.data_point.text
                pos = label.value
                # disambiguate homographs
                if token in self.homographs or token.lower() in self.homographs:  # This is really ineffective, but we need to check identity and lowercase, otherwise we won't find homographs at beginning of sentences if written in upper case
                    print("found homograph: ", token, "\t POS: ", pos)
                    wiki_pos = self.poet_to_wiktionary.get(pos, pos)
                    resolved = False
                   
                    # 'plus' is tricky and needs special treatment
                    if token == "plus" and wiki_pos == "adverbe":
                        # Wenn plus eine negative Bedeutung hat (d. h. es bedeutet ‘nicht(s) mehr’, ‘keine mehr’) sprechen wir das -s am Ende nicht aus.
                        if re.search(r"(\b(ne|non)\b)", text) or re.search(r"\bn(\’|\')", text):
                            # print("found negation")
                            pronunciation = "ply"
                        # Wenn auf plus ein Adjektiv oder ein Adverb folgt, das mit einem Konsonaten beginnt, sprechen wir das -s nicht aus, auch wenn die Bedeutung positiv ist.
                        elif i < len(sentence) and (labels[i+1].value in ["ADV", 'ADJ','ADJMS','ADJFS','ADJMP','ADJFP']) and (sentence[i+1].text[0].lower() in ["b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "q", "r", "s", "t", "v", "w", "x", "z"]):
                            print("plus before adjective or adverb")
                            print(sentence[i+1].text[0])
                            pronunciation = "ply"
                        # Wenn plus eine positive Bedeutung hat (d. h. es bedeutet ‘mehr’, ‘zusätzlich’), sprechen wir das -s am Ende aus.
                        else:
                            pronunciation = "plys" # in theory, there is also a difference between /plys/ and /plyz/ but maybe we can ignore this?
                        phones += self.phonemizer_backend.phonemize([chunk_to_phonemize], strip=True)[0]
                        phones += " " + pronunciation + " " 
                        chunk_to_phonemize = " "
                        continue # we're done with 'plus' move on to next token without checking anything else

                    # get candidates with correct pos tag
                    candidates = [entry for entry in self.homographs[token] if entry['pos'] == wiki_pos]
                    # print(candidates)

                    # no candidates were found for POS tag, we don't need to check anything further
                    if len(candidates) == 0:
                        chunk_to_phonemize += token
                        print(f"no matching candidates found for {token}: {pos}")
                        continue

                    # resolve if there are multiple pronunciations for one entry. For now, ignore lists
                    pronunciation_set = set(entry['pronunciation'] for entry in candidates if not type(entry['pronunciation']) == list)
                    if len(pronunciation_set) == 1:  # all entries have the same pronunciation, so we can just take it
                        pronunciation = pronunciation_set.pop()
                        print(f"All entries have the same pronunciation for {token}", pronunciation) 
                        resolved = True                
                    else: # TODO: needs further action
                        print("There are different pronunciations in the entries for ", token)

                        for entry in candidates:
                            if "pos_details" in entry and entry['pos_details'] == pos:
                                pronunciation = entry['pronunciation']
                                resolved = True
                                print(f"found pos details for {token} ({pos}): {pronunciation}")    
                                break # we found our match, no need to look further
                            elif "defult" in entry and entry['default'] == "True":
                                pronunciation == entry['pronunciation'] # found default pronunciation, but keep searching for matching pos_details
                                resolved = True
                                print(f"found default pronunciation for {token} ({pos}): {pronunciation}")
                    
                    # we found a homograph and could resolve it, so let's phonemize everything up to this point
                    if resolved == True:
                        chunk_to_phonemize += token # we add the homograph token and replace it later, because we don't want to lose liaisons etc.
                        phones += self.phonemizer_backend.phonemize([chunk_to_phonemize], strip=True)[0]
                        phones = phones.rsplit(" ", 1)[0] + " " + pronunciation + " " # remove espeak phonemes for homograph token and replace them with gold phonemes
                        chunk_to_phonemize = " "
                    else: # there is a homograph but we couldn't resolve it, add it to chunk and let espeak handle it when chunk is phonemized
                        chunk_to_phonemize += token + " "
                        print(f"Couldn't disambiguate homograph {token} ({pos}). Fall back on espeak.")
                else: # no homograph found
                    chunk_to_phonemize += token + " "

            phones += self.phonemizer_backend.phonemize([chunk_to_phonemize], strip=True)[0]  # add last part of phone string
        else:
            phones = self.phonemizer_backend.phonemize([utt], strip=True)[0]  # To use a different phonemizer, this is the only line that needs to be exchanged

        # Unfortunately tonal languages don't agree on the tone, most tonal
        # languages use different tones denoted by different numbering
        # systems. At this point in the script, it is attempted to unify
        # them all to the tones in the IPA standard.
        if self.g2p_lang == "vi":
            phones = phones.replace('1', "˧")
            phones = phones.replace('2', "˨˩")
            phones = phones.replace('ɜ', "˧˥")  # I'm fairly certain that this is a bug in espeak and ɜ is meant to be 3
            phones = phones.replace('3', "˧˥")  # I'm fairly certain that this is a bug in espeak and ɜ is meant to be 3
            phones = phones.replace('4', "˦˧˥")
            phones = phones.replace('5', "˧˩˧")
            phones = phones.replace('6', "˧˩ʔ˨")  # very weird tone, because the tone introduces another phoneme
            phones = phones.replace('7', "˧")

        return self.postprocess_phoneme_string(phones, for_feature_extraction, include_eos_symbol, for_plot_labels)

    def postprocess_phoneme_string(self, phoneme_string, for_feature_extraction, include_eos_symbol, for_plot_labels):
        """
        Takes as input a phoneme string and processes it to work best with the way we represent phonemes as featurevectors
        """
        replacements = [
            # punctuation in languages with non-latin script
            ("。", "."),
            ("，", ","),
            ("【", '"'),
            ("】", '"'),
            ("、", ","),
            ("‥", "…"),
            ("؟", "?"),
            ("،", ","),
            ("“", '"'),
            ("”", '"'),
            ("؛", ","),
            ("《", '"'),
            ("》", '"'),
            ("？", "?"),
            ("！", "!"),
            (" ：", ":"),
            (" ；", ";"),
            ("－", "-"),
            ("·", " "),
            # latin script punctuation
            ("/", " "),
            ("—", ""),
            ("...", "…"),
            ("\n", ", "),
            ("\t", " "),
            ("¡", ""),
            ("¿", ""),
            # unifying some phoneme representations
            ("ɫ", "l"),  # alveolopalatal
            ("ɚ", "ə"),
            ('ᵻ', 'ɨ'),
            ("ɧ", "ç"),  # velopalatal
            ("ɥ", "j"),  # labiopalatal
            ("ɬ", "s"),  # lateral
            ("ɮ", "z"),  # lateral
            ('ɺ', 'ɾ'),  # lateral
            ('\u02CC', ""),  # secondary stress
            ('\u030B', "˥"),
            ('\u0301', "˦"),
            ('\u0304', "˧"),
            ('\u0300', "˨"),
            ('\u030F', "˩"),
            ('\u0302', "⭨"),
            ('\u030C', "⭧"),
            ("꜖", "˩"),
            ("꜕", "˨"),
            ("꜔", "˧"),
            ("꜓", "˦"),
            ("꜒", "˥"),
            # symbols that indicate a pause or silence
            ('"', "~"),
            (" - ", "~"),
            ("-", ""),
            ("…", "."),
            (":", "~"),
            (";", "~"),
            (",", "~")  # make sure this remains the final one when adding new ones
        ]
        unsupported_ipa_characters = {'̹', '̙', '̞', '̯', '̤', '̪', '̩', '̠', '̟', 'ꜜ',
                                      '̬', '̽', 'ʰ', '|', '̝', '•', 'ˠ', '↘',
                                      '‖', '̰', '‿', 'ᷝ', '̈', 'ᷠ', '̜', 'ʷ', 'ʲ',
                                      '̚', '↗', 'ꜛ', '̻', '̥', 'ˁ', '̘', '͡', '̺'}
        # TODO support more of these. Problem: bridge over to aligner ID lookups after modifying the feature vector
        #  https://en.wikipedia.org/wiki/IPA_number
        for char in unsupported_ipa_characters:
            replacements.append((char, ""))

        if not for_feature_extraction:
            # in case we want to plot etc., we only need the segmental units, so we remove everything else.
            replacements = replacements + [
                ('\u02C8', ""),  # primary stress
                ('\u02D0', ""),  # lengthened
                ('\u02D1', ""),  # half-length
                ('\u0306', ""),  # shortened
                ("˥", ""),  # very high tone
                ("˦", ""),  # high tone
                ("˧", ""),  # mid tone
                ("˨", ""),  # low tone
                ("˩", ""),  # very low tone
                ('\u030C', ""),  # rising tone
                ('\u0302', ""),  # falling tone
                ('⭧', ""),  # rising
                ('⭨', ""),  # falling
                ('⮃', ""),  # dipping
                ('⮁', ""),  # peaking
                ('̃', ""),  # nasalizing
            ]
        for replacement in replacements:
            phoneme_string = phoneme_string.replace(replacement[0], replacement[1])
        phones = re.sub("~+", "~", phoneme_string)
        phones = re.sub(r"\s+", " ", phones)
        phones = re.sub(r"\.+", ".", phones)
        phones = phones.lstrip("~").rstrip("~")

        # peaking tones
        for peaking_perm in self.peaking_perms:
            phones = phones.replace(peaking_perm, "⮁".join(peaking_perm))
        # dipping tones
        for dipping_perm in self.dipping_perms:
            phones = phones.replace(dipping_perm, "⮃".join(dipping_perm))
        # rising tones
        for rising_perm in self.rising_perms:
            phones = phones.replace(rising_perm, "⭧".join(rising_perm))
        # falling tones
        for falling_perm in self.falling_perms:
            phones = phones.replace(falling_perm, "⭨".join(falling_perm))

        if self.add_silence_to_end:
            phones += "~"  # adding a silence in the end during inference produces more natural sounding prosody
        if include_eos_symbol:
            phones += "#"
        if not self.use_word_boundaries:
            phones = phones.replace(" ", "")
        if for_plot_labels:
            phones = phones.replace(" ", "|")

        phones = "~" + phones
        phones = re.sub("~+", "~", phones)

        return phones

    def text_vectors_to_id_sequence(self, text_vector):
        tokens = list()
        for vector in text_vector:
            if vector[get_feature_to_index_lookup()["word-boundary"]] == 0:
                # we don't include word boundaries when performing alignment, since they are not always present in audio.
                features = vector.cpu().numpy().tolist()
                if vector[get_feature_to_index_lookup()["vowel"]] == 1 and vector[get_feature_to_index_lookup()["nasal"]] == 1:
                    # for the sake of alignment, we ignore the difference between nasalized vowels and regular vowels
                    features[get_feature_to_index_lookup()["nasal"]] = 0
                features = features[13:]
                # the first 12 dimensions are for modifiers, so we ignore those when trying to find the phoneme in the ID lookup
                for phone in self.phone_to_vector:
                    if features == self.phone_to_vector[phone][13:]:
                        tokens.append(self.phone_to_id[phone])
                        # this is terribly inefficient, but it's fine
                        break
        return tokens


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


def french_spacing(text):
    text = text.replace("»", '"').replace("«", '"')
    for punc in ["!", ";", ":", ".", ",", "?"]:
        text = text.replace(f" {punc}", punc)
    return text


def convert_kanji_to_pinyin_mandarin(text):
    return " ".join([x[0] for x in pinyin(text)])


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
    elif language == "cmn":
        return torch.LongTensor([13])
    elif language == "vi":
        return torch.LongTensor([14])
    elif language == "uk":
        return torch.LongTensor([15])
    elif language == "fa":
        return torch.LongTensor([16])
    elif language == "pt-br":
        return torch.LongTensor([17])


def load_json_from_path(path):
    with open(path, "r", encoding="utf8") as f:
        obj = json.loads(f.read())
    return obj


if __name__ == '__main__':
    tf = ArticulatoryCombinedTextFrontend(language="fr")

    tf = ArticulatoryCombinedTextFrontend(language="en")
    tf.string_to_tensor("This is a complex sentence, it even has a pause! But can it do this? Nice.", view=True)

    tf = ArticulatoryCombinedTextFrontend(language="de")
    tf.string_to_tensor("Alles klar, jetzt testen wir einen deutschen Satz. Ich hoffe es gibt nicht mehr viele unspezifizierte Phoneme.", view=True)

    tf = ArticulatoryCombinedTextFrontend(language="cmn")
    tf.string_to_tensor("这是一个复杂的句子，它甚至包含一个停顿。", view=True)
    tf.string_to_tensor("李绅 《悯农》 锄禾日当午， 汗滴禾下土。 谁知盘中餐， 粒粒皆辛苦。", view=True)
    tf.string_to_tensor("巴 拔 把 爸 吧", view=True)

    tf = ArticulatoryCombinedTextFrontend(language="vi")
    tf.string_to_tensor("Xin chào thế giới, quả là một ngày tốt lành để học nói tiếng Việt!", view=True)
    tf.string_to_tensor("ba bà bá bạ bả bã", view=True)

    tf = ArticulatoryCombinedTextFrontend(language="fr")
    tf.string_to_tensor("Je ne te fais pas un dessin.", view=True)
    print(tf.get_phone_string("Je ne te fais pas un dessin."))
    print(tf.string_to_tensor("un", view=True))
