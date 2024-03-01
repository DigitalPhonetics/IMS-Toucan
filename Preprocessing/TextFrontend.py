# -*- coding: utf-8 -*-


import json
import re

import torch
from dragonmapper.transcriptions import pinyin_to_ipa
from phonemizer.backend import EspeakBackend
from pypinyin import pinyin

from Preprocessing.articulatory_features import generate_feature_table
from Preprocessing.articulatory_features import get_feature_to_index_lookup
from Preprocessing.articulatory_features import get_phone_to_id


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

        # this locks the device, so it has to happen here and not at the top
        from transphone.g2p import read_g2p

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

        if language == "eng":
            self.g2p_lang = "en-us"  # English as spoken in USA
            self.expand_abbreviations = english_text_expansion
            self.phonemizer = "espeak"

        elif language == "deu":
            self.g2p_lang = "de"  # German
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "ell":
            self.g2p_lang = "el"  # Greek
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "spa":
            self.g2p_lang = "es"  # Spanish
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "fin":
            self.g2p_lang = "fi"  # Finnish
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "rus":
            self.g2p_lang = "ru"  # Russian
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "hun":
            self.g2p_lang = "hu"  # Hungarian
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "nld":
            self.g2p_lang = "nl"  # Dutch
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "fra":
            self.g2p_lang = "fr-fr"  # French
            self.expand_abbreviations = remove_french_spacing
            self.phonemizer = "espeak"

        elif language == "ita":
            self.g2p_lang = "it"  # Italian
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "por":
            self.g2p_lang = "pt"  # Portuguese
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "pol":
            self.g2p_lang = "pl"  # Polish
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "cmn":
            self.g2p_lang = "cmn"  # Mandarin
            self.expand_abbreviations = convert_kanji_to_pinyin_mandarin
            self.phonemizer = "dragonmapper"

        elif language == "vie":
            self.g2p_lang = "vi"  # Northern Vietnamese
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "ukr":
            self.g2p_lang = "uk"  # Ukrainian
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "pes":
            self.g2p_lang = "fa"  # Farsi
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "afr":
            self.g2p_lang = "af"  # Afrikaans
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "aln":
            self.g2p_lang = "sq"  # Albanian
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "amh":
            self.g2p_lang = "am"  # Amharic
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "arb":
            self.g2p_lang = "ar"  # Arabic
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "arg":
            self.g2p_lang = "an"  # Aragonese
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "hye":
            self.g2p_lang = "hy"  # East Armenian
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "hyw":
            self.g2p_lang = "hyw"  # West Armenian
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "azj":
            self.g2p_lang = "az"  # Azerbaijani
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "bak":
            self.g2p_lang = "ba"  # Bashkir
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "eus":
            self.g2p_lang = "eu"  # Basque
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "bel":
            self.g2p_lang = "be"  # Belarusian
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "ben":
            self.g2p_lang = "bn"  # Bengali
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "bpy":
            self.g2p_lang = "bpy"  # Bishnupriya Manipuri
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "bos":
            self.g2p_lang = "bs"  # Bosnian
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "bul":
            self.g2p_lang = "bg"  # Bulgarian
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "mya":
            self.g2p_lang = "my"  # Burmese
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "chr":
            self.g2p_lang = "chr"  # Cherokee
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "yue":
            self.g2p_lang = "yue"  # Chinese	Cantonese
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "hak":
            self.g2p_lang = "hak"  # Chinese	Hakka
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "haw":
            self.g2p_lang = "haw"  # Hawaiian
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "hrv":
            self.g2p_lang = "hr"  # Croatian
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "ces":
            self.g2p_lang = "cs"  # Czech
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "dan":
            self.g2p_lang = "da"  # Danish
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "ekk":
            self.g2p_lang = "et"  # Estonian
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "gle":
            self.g2p_lang = "ga"  # Gaelic	Irish
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "gla":
            self.g2p_lang = "gd"  # Gaelic	Scottish
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "kat":
            self.g2p_lang = "ka"  # Georgian
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "kal":
            self.g2p_lang = "kl"  # Greenlandic
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "guj":
            self.g2p_lang = "gu"  # Gujarati
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "heb":
            self.g2p_lang = "he"  # Hebrew
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "hin":
            self.g2p_lang = "hi"  # Hindi
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "isl":
            self.g2p_lang = "is"  # Icelandic
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "ind":
            self.g2p_lang = "id"  # Indonesian
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "jpn":
            self.g2p_lang = "ja"  # Japanese
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "kan":
            self.g2p_lang = "kn"  # Kannada
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "knn":
            self.g2p_lang = "kok"  # Konkani
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "kor":
            self.g2p_lang = "ko"  # Korean
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "ckb":
            self.g2p_lang = "ku"  # Kurdish
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "kaz":
            self.g2p_lang = "kk"  # Kazakh
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "kir":
            self.g2p_lang = "ky"  # Kyrgyz
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "lat":
            self.g2p_lang = "la"  # Latin
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "ltz":
            self.g2p_lang = "lb"  # Luxembourgish
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "lvs":
            self.g2p_lang = "lv"  # Latvian
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "lit":
            self.g2p_lang = "lt"  # Lithuanian
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "mri":
            self.g2p_lang = "mi"  # Māori
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "mkd":
            self.g2p_lang = "mk"  # Macedonian
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "zlm":
            self.g2p_lang = "ms"  # Malay
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "mal":
            self.g2p_lang = "ml"  # Malayalam
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "mlt":
            self.g2p_lang = "mt"  # Maltese
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "mar":
            self.g2p_lang = "mr"  # Marathi
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "nci":
            self.g2p_lang = "nci"  # Nahuatl
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "npi":
            self.g2p_lang = "ne"  # Nepali
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "nob":
            self.g2p_lang = "nb"  # Norwegian Bokmål
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "nog":
            self.g2p_lang = "nog"  # Nogai
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "ory":
            self.g2p_lang = "or"  # Oriya
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "gaz":
            self.g2p_lang = "om"  # Oromo
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "pap":
            self.g2p_lang = "pap"  # Papiamento
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "pan":
            self.g2p_lang = "pa"  # Punjabi
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "ron":
            self.g2p_lang = "ro"  # Romanian
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "lav":
            self.g2p_lang = "ru-lv"  # Russian	Latvia
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "srp":
            self.g2p_lang = "sr"  # Serbian
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "tsn":
            self.g2p_lang = "tn"  # Setswana
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "snd":
            self.g2p_lang = "sd"  # Sindhi
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "slk":
            self.g2p_lang = "sk"  # Slovak
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "slv":
            self.g2p_lang = "sl"  # Slovenian
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "smj":
            self.g2p_lang = "smj"  # Lule Saami
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "swh":
            self.g2p_lang = "sw"  # Swahili
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "swe":
            self.g2p_lang = "sv"  # Swedish
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "tam":
            self.g2p_lang = "ta"  # Tamil
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "tha":
            self.g2p_lang = "th"  # Thai
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "tuk":
            self.g2p_lang = "tk"  # Turkmen
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "tat":
            self.g2p_lang = "tt"  # Tatar
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "tel":
            self.g2p_lang = "te"  # Telugu
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "tur":
            self.g2p_lang = "tr"  # Turkish
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "uig":
            self.g2p_lang = "ug"  # Uyghur
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "urd":
            self.g2p_lang = "ur"  # Urdu
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "uzn":
            self.g2p_lang = "uz"  # Uzbek
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        elif language == "cym":
            self.g2p_lang = "cy"  # Welsh
            self.expand_abbreviations = lambda x: x
            self.phonemizer = "espeak"

        else:
            # blanket solution for the rest
            self.g2p_lang = language
            self.phonemizer = "transphone"
            self.expand_abbreviations = lambda x: x
            self.transphone = read_g2p()

        # remember to also update get_language_id() below when adding something here, as well as the get_example_sentence function

        if self.phonemizer == "espeak":
            self.phonemizer_backend = EspeakBackend(language=self.g2p_lang,
                                                    punctuation_marks=';:,.!?¡¿—…"«»“”~/。【】、‥،؟“”؛',
                                                    preserve_punctuation=True,
                                                    language_switch='remove-flags',
                                                    with_stress=self.use_stress)

        self.phone_to_vector = generate_feature_table()
        self.phone_to_id = get_phone_to_id()
        self.id_to_phone = {v: k for k, v in self.phone_to_id.items()}
        self.text_vector_to_phone_cache = dict()

    @staticmethod
    def get_example_sentence(lang):
        if lang == "eng":
            return "This is a complex sentence, it even has a pause!"
        elif lang == "deu":
            return "Dies ist ein komplexer Satz, er hat sogar eine Pause!"
        elif lang == "ell":
            return "Αυτή είναι μια σύνθετη πρόταση, έχει ακόμη και παύση!"
        elif lang == "spa":
            return "Esta es una oración compleja, ¡incluso tiene una pausa!"
        elif lang == "fin":
            return "Tämä on monimutkainen lause, sillä on jopa tauko!"
        elif lang == "rus":
            return "Это сложное предложение, в нем даже есть пауза!"
        elif lang == "hun":
            return "Ez egy összetett mondat, még szünet is van benne!"
        elif lang == "nld":
            return "Dit is een complexe zin, er zit zelfs een pauze in!"
        elif lang == "fra":
            return "C'est une phrase complexe, elle a même une pause !"
        elif lang == "por":
            return "Esta é uma frase complexa, tem até uma pausa!"
        elif lang == "pol":
            return "To jest zdanie złożone, ma nawet pauzę!"
        elif lang == "ita":
            return "Questa è una frase complessa, ha anche una pausa!"
        elif lang == "cmn":
            return "这是一个复杂的句子，它甚至包含一个停顿。"
        elif lang == "vie":
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
            elif char == "̧":
                # palatalized
                phones_vector[-1][get_feature_to_index_lookup()["palatal"]] = 1
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

    def get_phone_string(self, text, include_eos_symbol=True, for_feature_extraction=False, for_plot_labels=False):
        # expand abbreviations
        utt = self.expand_abbreviations(text)

        # convert the graphemes to phonemes here
        if self.phonemizer == "espeak":
            try:
                phones = self.phonemizer_backend.phonemize([utt], strip=True)[0]  # To use a different phonemizer, this is the only line that needs to be exchanged
            except:
                print(f"There was an error with espeak. \nFalling back to transphone.\nSentence: {utt} \nLanguage {self.g2p_lang}")
                from transphone.g2p import read_g2p
                self.g2p_lang = self.language
                self.phonemizer = "transphone"
                self.expand_abbreviations = lambda x: x
                self.transphone = read_g2p()
                return self.get_phone_string(text, include_eos_symbol, for_feature_extraction, for_plot_labels)
        elif self.phonemizer == "transphone":
            replacements = [
                # punctuation in languages with non-latin script
                ("。", "~"),
                ("，", "~"),
                ("【", '~'),
                ("】", '~'),
                ("、", "~"),
                ("‥", "~"),
                ("؟", "~"),
                ("،", "~"),
                ("“", '~'),
                ("”", '~'),
                ("؛", "~"),
                ("《", '~'),
                ("》", '~'),
                ("？", "~"),
                ("！", "~"),
                (" ：", "~"),
                (" ；", "~"),
                ("－", "~"),
                ("·", " "),
                # symbols that indicate a pause or silence
                ('"', "~"),
                (" - ", "~ "),
                ("- ", "~ "),
                ("-", ""),
                ("…", "~"),
                (":", "~"),
                (";", "~"),
                (",", "~")  # make sure this remains the final one when adding new ones
            ]
            for replacement in replacements:
                utt = utt.replace(replacement[0], replacement[1])
            utt = re.sub("~+", "~", utt)
            utt = re.sub(r"\s+", " ", utt)
            utt = re.sub(r"\.+", ".", utt)
            chunk_list = list()
            for chunk in utt.split("~"):
                # unfortunately the transphone tokenizer is not suited for any languages besides English it seems
                # this is not much better, but maybe a little.
                word_list = list()
                for word_by_whitespace in chunk.split():
                    word_list.append(self.transphone.inference(word_by_whitespace, self.g2p_lang))
                chunk_list.append(" ".join(["".join(word) for word in word_list]))
            phones = "~ ".join(chunk_list)
        elif self.phonemizer == "dragonmapper":
            phones = pinyin_to_ipa(utt)

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
            phones = phones.replace('6', "˧˩˨ʔ")  # very weird tone, because the tone introduces another phoneme
            phones = phones.replace('7', "˧")
        # TODO add more of this handling for more tonal languages
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
            ("«", '"'),
            ("»", '"'),
            # unifying some phoneme representations
            ("ɫ", "l"),  # alveolopalatal
            ("ɚ", "ə"),
            ('ᵻ', 'ɨ'),
            ("ɧ", "ç"),  # velopalatal
            ("ɥ", "j"),  # labiopalatal
            ("ɬ", "s"),  # lateral
            ("ɮ", "z"),  # lateral
            ('ɺ', 'ɾ'),  # lateral
            ('ʲ', 'j'),  # decomposed palatalization
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
            (" - ", "~ "),
            ("- ", "~ "),
            ("-", ""),
            ("…", "."),
            (":", "~"),
            (";", "~"),
            (",", "~")  # make sure this remains the final one when adding new ones
        ]
        unsupported_ipa_characters = {'̹', '̙', '̞', '̯', '̤', '̪', '̩', '̠', '̟', 'ꜜ',
                                      '̬', '̽', 'ʰ', '|', '̝', '•', 'ˠ', '↘',
                                      '‖', '̰', '‿', 'ᷝ', '̈', 'ᷠ', '̜', 'ʷ',
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
                immutable_vector = tuple(features)
                if immutable_vector in self.text_vector_to_phone_cache:
                    tokens.append(self.phone_to_id[self.text_vector_to_phone_cache[immutable_vector]])
                    continue
                if vector[get_feature_to_index_lookup()["vowel"]] == 1 and vector[get_feature_to_index_lookup()["nasal"]] == 1:
                    # for the sake of alignment, we ignore the difference between nasalized vowels and regular vowels
                    features[get_feature_to_index_lookup()["nasal"]] = 0
                features = features[13:]
                # the first 12 dimensions are for modifiers, so we ignore those when trying to find the phoneme in the ID lookup
                for phone in self.phone_to_vector:
                    if features == self.phone_to_vector[phone][13:]:
                        tokens.append(self.phone_to_id[phone])
                        self.text_vector_to_phone_cache[immutable_vector] = phone
                        # this is terribly inefficient, but it's fine, since we're building a cache over time that makes this instant
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


def remove_french_spacing(text):
    text = text.replace(" »", '"').replace("« ", '"')
    for punc in ["!", ";", ":", ".", ",", "?", "-"]:
        text = text.replace(f" {punc}", punc)
    return text


def convert_kanji_to_pinyin_mandarin(text):
    return " ".join([x[0] for x in pinyin(text)])


def get_language_id(language):
    try:
        iso_codes_to_ids = load_json_from_path("Preprocessing/multilinguality/iso_lookup.json")[-1]
    except FileNotFoundError:
        iso_codes_to_ids = load_json_from_path("multilinguality/iso_lookup.json")[-1]
    if language not in iso_codes_to_ids:
        print("Please specify the language as ISO 639-2 code (https://en.wikipedia.org/wiki/List_of_ISO_639-2_codes)")
        return None
    return torch.LongTensor([iso_codes_to_ids[language]])


def load_json_from_path(path):
    with open(path, "r", encoding="utf8") as f:
        obj = json.loads(f.read())
    return obj


if __name__ == '__main__':
    tf = ArticulatoryCombinedTextFrontend(language="eng")
    tf.string_to_tensor("This is a complex sentence, it even has a pause! But can it do this? Nice.", view=True)

    tf = ArticulatoryCombinedTextFrontend(language="deu")
    tf.string_to_tensor("Alles klar, jetzt testen wir einen deutschen Satz. Ich hoffe es gibt nicht mehr viele unspezifizierte Phoneme.", view=True)

    tf = ArticulatoryCombinedTextFrontend(language="cmn")
    tf.string_to_tensor("这是一个复杂的句子，它甚至包含一个停顿。", view=True)
    tf.string_to_tensor("李绅 《悯农》 锄禾日当午， 汗滴禾下土。 谁知盘中餐， 粒粒皆辛苦。", view=True)
    tf.string_to_tensor("巴 拔 把 爸 吧", view=True)

    tf = ArticulatoryCombinedTextFrontend(language="vie")
    tf.string_to_tensor("Xin chào thế giới, quả là một ngày tốt lành để học nói tiếng Việt!", view=True)
    tf.string_to_tensor("ba bà bá bạ bả bã", view=True)

    tf = ArticulatoryCombinedTextFrontend(language="fra")
    tf.string_to_tensor("Je ne te fais pas un dessin.", view=True)
    print(tf.get_phone_string("Je ne te fais pas un dessin."))

    tf = ArticulatoryCombinedTextFrontend(language="acr")
    tf.string_to_tensor("I don't know this language, but this is just a dummy anyway.", view=True)
    print(tf.get_phone_string("I don't know this language, but this is just a dummy anyway."))

    print(get_language_id("eng"))
