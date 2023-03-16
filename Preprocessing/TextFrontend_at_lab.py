# -*- coding: utf-8 -*-


import re
import sys
import os
import torch
import phonemizer
from phonemizer.backend import EspeakBackend
from phonemizer.backend import FestivalBackend
from pypinyin import pinyin

from Preprocessing.articulatory_features import generate_feature_table
from Preprocessing.articulatory_features import get_phone_to_id


class ArticulatoryCombinedTextFrontend:

    def __init__(self,
                 language,
                 use_explicit_eos=True,
                 use_lexical_stress=True,
                 silent=True,
                 allow_unknown=False,
                 add_silence_to_end=True,
                 path_to_sampa_mapping_list="Preprocessing/sampa_to_ipa_punct.txt"):
        """
        Mostly preparing ID lookups
        """
        self.allow_unknown = allow_unknown
        self.use_explicit_eos = use_explicit_eos
        self.use_stress = use_lexical_stress
        self.add_silence_to_end = add_silence_to_end
        self.sampa_to_ipa_dict = dict()
        
        #FestivalBackend.set_executable("/data/vokquant/CSTR-HTSVoice-Library-ver0.99/festival/bin/festival")
        FestivalBackend.set_festival_path("/data/vokquant/CSTR-HTSVoice-Library-ver0.99/festival/bin/festival")

        with open(path_to_sampa_mapping_list, "r", encoding='utf8') as f:
            sampa_to_ipa = f.read()
        sampa_to_ipa_list = sampa_to_ipa.split("\n")
        for pair in sampa_to_ipa_list:
            if pair.strip() != "":
                #print(pair)
                self.sampa_to_ipa_dict[pair.split(" ")[0]] = pair.split(" ")[1]

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

        elif language == "cmn":
            self.g2p_lang = "cmn-latn-pinyin"  # in older versions of espeak this shorthand was zh
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

        elif language == "chr":
            self.g2p_lang = "chr"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Cherokee Text-Frontend")
                
        elif language == "at-lab":
            self.g2p_lang = "at-lab"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created an Austrian German Label Text-Frontend")

        elif language == "at":
            self.g2p_lang = "at"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created an Austrian German Text-Frontend")

        elif language == "vd":
            self.g2p_lang = "vd"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Viennese Text-Frontend")

        # remember to also update get_language_id() below when adding something here

        else:
            print("Language not supported yet")
            sys.exit()

        if self.g2p_lang=="at":
            print("g2p_lang: 'at' --> using festival")
        elif self.g2p_lang=="vd":
            print("g2p_lang: 'vd' --> using festival")
        elif self.g2p_lang=="at-lab":
            print("g2p_lang: 'at-lab' --> using labels from labelfiles")
            #self.phonemizer_backend = FestivalBackend(language=self.g2p_lang,
            #                                          punctuation_marks=';:,.!?¡¿—…"«»“”~/。【】、‥،؟“”؛',
            #                                          preserve_punctuation=False)
        else:
            print("use espeak")
            self.phonemizer_backend = EspeakBackend(language=self.g2p_lang,
                                                    punctuation_marks=';:,.!?¡¿—…"«»“”~/。【】、‥،؟“”؛',
                                                    preserve_punctuation=True,
                                                    language_switch='remove-flags',
                                                    with_stress=self.use_stress)
                                         
        self.phone_to_vector = generate_feature_table()
        self.phone_to_id = get_phone_to_id()
        self.id_to_phone = {v: k for k, v in self.phone_to_id.items()}

    def string_to_tensor(self, text, view=True, device="cpu", handle_missing=True, input_phonemes=False, path_to_wavfile=""):
        """
        Fixes unicode errors, expands some abbreviations,
        turns graphemes into phonemes and then vectorizes
        the sequence as articulatory features
        """
        if input_phonemes:
            phones = text
        else:
            #print("text (string_to_tensor):")
            #print(text)
            phones = self.get_phone_string(text=text, include_eos_symbol=True, for_feature_extraction=True, path_to_wavfile=path_to_wavfile)
        if view:
            print("Phonemes (string_to_tensor) look like this: \n{}\n".format(phones))
        phones_vector = list()
        # turn into numeric vectors
        stressed_flag = False

        for char in phones:
            if char == '\u02C8':
                # primary stress
                # affects following phoneme
                stressed_flag = True
            elif char == '\u02D0':
                # lengthened
                # affects previous phoneme
                phones_vector[-1][8] = 1
            elif char == '\u02D1':
                # half length
                # affects previous phoneme
                phones_vector[-1][9] = 1
            elif char == '\u0306':
                # shortened
                # affects previous phoneme
                phones_vector[-1][10] = 1
            elif char == "˥":
                # very high tone
                # affects previous phoneme
                phones_vector[-1][1] = 1
            elif char == "˦":
                # high tone
                # affects previous phoneme
                phones_vector[-1][2] = 1
            elif char == "˧":
                # mid tone
                # affects previous phoneme
                phones_vector[-1][3] = 1
            elif char == "˨":
                # low tone
                # affects previous phoneme
                phones_vector[-1][4] = 1
            elif char == "˩":
                # very low tone
                # affects previous phoneme
                phones_vector[-1][5] = 1
            elif char == '\u030C':
                # rising tone
                # affects previous phoneme
                phones_vector[-1][6] = 1
            elif char == '\u0302':
                # falling tone
                # affects previous phoneme
                phones_vector[-1][7] = 1
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
                    phones_vector[-1][0] = 1

        return torch.Tensor(phones_vector, device=device)

    def phonemize_from_labelfile(self, text, path_to_wavfile, include_eos_symbol=True):
        if os.path.exists(path_to_wavfile):
            print(path_to_wavfile)
            head, tail = os.path.split(path_to_wavfile)
            labelfile=tail.replace(".wav",".lab")
            print(labelfile)
            sampa_phones=[]
            phones=""
            with open(os.path.join(head.replace("aridialect_wav16000","aridialect_labels"),labelfile), encoding="utf8") as f:
                labels = f.read()
            label_lines = labels.split("\n")
            for line in label_lines:
                if line.strip() != "":
                    sampa_phones.append(line[line.find("-")+1:line.find("+")])
            #print(sampa_phones)
            phones = self.sampa_to_ipa(sampa_phones)
            #if self.strip_silence:
            #    phones = phones.lstrip("~").rstrip("~")
            #preserve final punctuation
            #print(text[len(text)-1])
            #if ';:,.!?¡¿—…"«»“”~/'.find(text[len(text)-1].strip())!=-1:
            #    phones = phones + text[len(text)-1].strip()
            #print(phones)
            return phones
        else:
            print("path does not exist: "+path_to_wavfile)

    def sampa_to_ipa(self, sampa_phones):
        ipa_phones = ""
        for p in sampa_phones:
          if p not in ';:,.!?¡¿—…"«»“”~/':
             ipa_phones = ipa_phones+self.sampa_to_ipa_dict[p]

        return ipa_phones.replace(";", ",").replace("/", " ") \
                .replace(":", ",").replace('"', ",").replace("-", ",").replace("-", ",").replace("\n", " ") \
                .replace("\t", " ").replace("¡", "").replace("¿", "").replace(",", "~")

    def get_phone_string(self, text, include_eos_symbol=True, for_feature_extraction=False, for_plot_labels=False, path_to_wavfile=""):
        # expand abbreviations
        #print("get_phone_string 'text': \n"+ text)
        utt = self.expand_abbreviations(text)
        
        # phonemize
        if self.g2p_lang=="at" or self.g2p_lang=="vd":
            #phones = self.phonemizer_backend.phonemize([utt], strip=True)[0]
            #phones = self.phonemizer_backend.phonemize([utt], strip=True)
            phones = phonemizer.phonemize(text=utt,
                                          backend="festival",
                                          language=self.g2p_lang,
                                          preserve_punctuation=True,
                                          strip=False,
                                          punctuation_marks=';:,.!?¡¿—…"«»“”~/'
                                         )
            #print("here: ")
            #print(type(phones))
        elif self.g2p_lang=="at-lab":
            phones = self.phonemize_from_labelfile(text=utt, path_to_wavfile=path_to_wavfile, include_eos_symbol=False)
        else:
            phones = self.phonemizer_backend.phonemize([utt], strip=True)[0]
        #
        #print(phones)

        # Unfortunately tonal languages don't agree on the tone, most tonal
        # languages use different tones denoted by different numbering
        # systems. At this point in the script, it is attempted to unify
        # them all to the tones in the IPA standard.
        if self.g2p_lang == "cmn-latn-pinyin" or self.g2p_lang == "cmn":
            phones = phones.replace(".", "")  # no idea why espeak puts dots everywhere for Chinese
            phones = phones.replace('1', "˥")
            phones = phones.replace('2', "˧\u030C")
            phones = phones.replace('ɜ', "˨\u0302\u030C")  # I'm fairly certain that this is a bug in espeak and ɜ is meant to be 3
            phones = phones.replace('3', "˨\u0302\u030C")  # I'm fairly certain that this is a bug in espeak and ɜ is meant to be 3
            phones = phones.replace('4', "˦\u0302")
            phones = phones.replace('5', "˧")
            phones = phones.replace('0', "˧")
        if self.g2p_lang == "vi":
            phones = phones.replace('1', "˧")
            phones = phones.replace('2', "˩\u0302")
            phones = phones.replace('ɜ', "˧\u030C")  # I'm fairly certain that this is a bug in espeak and ɜ is meant to be 3
            phones = phones.replace('3', "˧\u030C")  # I'm fairly certain that this is a bug in espeak and ɜ is meant to be 3
            phones = phones.replace('4', "˧\u0302\u030C")
            phones = phones.replace('5', "˧\u030C")
            phones = phones.replace('6', "˧\u0302")
            phones = phones.replace('7', "˧")
        replacements = [
            # punctuation in languages with non-latin script
            ("。", "."),
            ("【", '"'),
            ("】", '"'),
            ("、", ","),
            ("‥", "…"),
            ("؟", "?"),
            ("،", ","),
            ("“", '"'),
            ("”", '"'),
            ("؛", ","),
            # latin script punctuation
            ("/", " "),
            ("—", ""),
            ("...", "…"),
            #("\n", " "),
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
            # Lorenz sampa to IPA
            ('schwa_',"ə"),
            ('gsth_',"ɡ"),
            ('bsth_',"b"),
            ('dsth_',"d"),
            ('P2h6_',"øɐ"),
            ('P9hn_',"œn"),
            ('P9P2_',"œø"),
            ('P3hn_',"ɛ"),
            ('sil_',"~"),
            ('aAN_',"aɑ"),
            ('aeN_',"aeŋ"),
            ('aen_',"aen"),
            ('Ah6_',"ɑɐ"),
            ('ah6_',"aɐ"),
            ('Ahn_',"ɑn"),
            ('AhN_',"ɑŋ"),
            ('ahn_',"an"),
            ('ahN_',"aŋ"),
            ('aO1_',"aɔɶ"),
            ('ao1_',"aoɶ"),
            ('aoN_',"aoŋ"),
            ('yh6_',"ʏɐ"),
            ('yP6_',"ʏɐ"),
            ('UP6_',"ʊɐ"),
            ('uP6_',"uɐ"),
            ('EP6_',"e"),
            ('eP6_',"eɐ"),
            ('EeN_',"ɛeŋ"),
            ('Eh6_',"ɛn"),
            ('Ehn_',"ɛn"),
            ('EhN_',"ɛŋ"),
            ('ih6_',"iɐ"),
            ('ihn_',"in"),
            ('ihN_',"iŋ"),
            ('kch_',"kx"),
            ('e~:_',"e"),
            ('iP6_',"iɐ"),
            ('oaN_',"oaŋ"),
            ('OaN_',"ɔaŋ"),
            ('Oan_',"ɔan"),
            ('oan_',"oan"),
            ('Oh6_',"ɔɐ"),
            ('oh6_',"oɐ"),
            ('Ohn_',"ɔn"),
            ('ohn_',"on"),
            ('P6N_',"ɐŋ"),
            ('P6n_',"ɐn"),
            ('P6O_',"ɐɔ"),
            ('P6U_',"ɐʊ"),
            ('P96_',"œɐ"),
            ('P9e_',"œe"),
            ('P9h_',"œ"),
            ('pau_',"~"),
            ('OP6_',"ɔɐ"),
            ('P1h_',"ɶ"),
            ('P2h_',"ø"),
            ('P3h_',"ɛ"),
            ('P1:_',"ɶ"),
            ('uh6_',"uɐ"),
            ('Uh6_',"ʊɐ"),
            ('A6_',"ɑɐ"),
            ('a6_',"aɐ"),
            ('aA_',"aɑ"),
            ('ae_',"ae"),
            ('aE_',"aɛ"),
            ('ah_',"a"),
            ('Ah_',"ɑ"),
            ('AI_',"ɑɪ"),
            ('aI_',"aɪ"),
            ('AN_',"ɑŋ"),
            ('aN_',"aŋ"),
            ('An_',"ɑn"),
            ('an_',"an"),
            ('ao_',"ao"),
            ('aO_',"aɔ"),
            ('aU_',"aʊ"),
            ('Y6_',"ʏɐ"),
            ('yh_',"ʏ"),
            ('bf_',"bf"),
            ('ch_',"x"),
            ('dF_',"d"),
            ('E6_',"ɛɐ"),
            ('ea_',"ea"),
            ('Ea_',"ɛa"),
            ('eE_',"eɛ"),
            ('Ee_',"ɛe"),
            ('eh_',"e"),
            ('Eh_',"ɛ"),
            ('Ei_',"ɛi"),
            ('EN_',"ɛŋ"),
            ('En_',"ɛn"),
            ('GS_',"ʔ"),
            ('I6_',"ɪɐ"),
            ('i6_',"iɐ"),
            ('iE_',"iɛ"),
            ('ih_',"i"),
            ('Ii_',"ɪi"),
            ('iN_',"iŋ"),
            ('in_',"in"),
            ('iV_',"i"),
            ('kH_',"kɥ"),
            ('ks_',"ks"),
            ('ll_',"l"),
            ('ml_',"m"),
            ('Nl_',"ŋ"),
            ('nl_',"n"),
            ('O6_',"ɔɐ"),
            ('Oa_',"ɔ"),
            ('oa_',"o"),
            ('Oe_',"ɔe"),
            ('OE_',"ɔɛ"),
            ('oe_',"oe"),
            ('Oh_',"ɔ"),
            ('oh_',"o"),
            ('oI_',"oɪ"),
            ('oi_',"oi"),
            ('ON_',"ɔŋ"),
            ('On_',"ɔn"),
            ('Oo_',"ɔo"),
            ('OU_',"ɔʊ"),
            ('OY_',"ɔʏ"),
            ('P2_',"ø"),
            ('P6_',"ɐ"),
            ('P9_',"œ"),
            ('pH_',"pɥ"),
            ('Qh_',"ɒ"),
            ('RX_',"ʀχ"),
            ('sh_',"s"),
            ('tH_',"tɥ"),
            ('tS_',"tʃ"),
            ('ts_',"ts"),
            ('U6_',"ʊɐ"),
            ('ua_',"u"),
            ('ue_',"u"),
            ('uh_',"u"),
            ('Ui_',"ʊi"),
            ('ui_',"ui"),
            ('uI_',"uɪ"),
            ('uN_',"uŋ"),
            ('Uu_',"ʊu"),
            ('a_',"a"),
            ('B_',"β"),
            ('b_',"b"),
            ('E_',"ɛ"),
            ('C_',"ç"),
            ('D_',"ð"),
            ('d_',"d"),
            ('e_',"e"),
            ('f_',"f"),
            ('G_',"ɣ"),
            ('g_',"ɡ"),
            ('h_',"h"),
            ('I_',"ɪ"),
            ('i_',"i"),
            ('j_',"j"),
            ('k_',"k"),
            ('L_',"ʎ"),
            ('l_',"l"),
            ('m_',"m"),
            ('N_',"ŋ"),
            ('n_',"n"),
            ('O_',"ɔ"),
            ('o_',"o"),
            ('p_',"p"),
            ('R_',"ʀ"),
            ('r_',"r"),
            ('S_',"ʃ"),
            ('s_',"s"),
            ('t_',"t"),
            ('U_',"ʊ"),
            ('u_',"u"),
            ('v_',"v"),
            ('Y_',"ʏ"),
            ('y_',"y"),
            ('Z_',"ʒ"),
            ('z_',"z"),
            # symbols that indicate a pause or silence
            ('"', "~"),
            ("-", "~"),
            ("-", "~"),
            ("…", "."),
            (":", "~"),
            (";", "~"),
            (",", "~")  # make sure this remains the final one when adding new ones
        ]
        unsupported_ipa_characters = {'̹', '̙', '̞', '̯', '̤', '̪', '̩', '̠', '̟', 'ꜜ',
                                      '̃', '̬', '̽', 'ʰ', '|', '̝', '•', 'ˠ', '↘',
                                      '‖', '̰', '‿', 'ᷝ', '̈', 'ᷠ', '̜', 'ʷ', 'ʲ',
                                      '̚', '↗', 'ꜛ', '̻', '̥', 'ˁ', '̘', '͡', '̺'}
        for char in unsupported_ipa_characters:
            replacements.append((char, ""))

        if not for_feature_extraction:
            # in case we want to plot etc., we only need the segmental units, so we remove everything else.
            replacements = replacements + [
                ('\u02C8', ""),  # primary stress
                ('\u02D0', ""),  # lengthened
                ('\u02D1', ""),  # half length
                ('\u0306', ""),  # shortened
                ("˥", ""),  # very high tone
                ("˦", ""),  # high tone
                ("˧", ""),  # mid tone
                ("˨", ""),  # low tone
                ("˩", ""),  # very low tone
                ('\u030C', ""),  # rising tone
                ('\u0302', "")  # falling tone
            ]
        for replacement in replacements:
            phones = phones.replace(replacement[0], replacement[1])
        phones = re.sub("~+", "~", phones)
        phones = re.sub(r"\s+", " ", phones)
        phones = re.sub(r"\.+", ".", phones)
        phones = phones.replace(" ~", "~").replace(" .", ".").replace(" ?", "?").replace(" !", "!")
        phones = phones.lstrip("~").rstrip("~")


        if self.add_silence_to_end:
            phones += "~"  # adding a silence in the end during inference produces more natural sounding prosody
        if include_eos_symbol:
            phones += "#"

        if for_plot_labels:
            phones = phones.replace(" ", "|")

        phones = "~" + phones
        phones = re.sub("~+", "~", phones)
        #print("finally, IPA phones look like this:")
        #print(phones)
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


def convert_kanji_to_pinyin_mandarin(text):
    # somehow the phonemizer looses the tone information, but
    # after the conversion to pinyin it is still there. Maybe
    # we need a better conversion from pinyin to IPA that
    # includes tone symbols if espeak-ng doesn't do a good job
    # on this.
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
    elif language == "chr":
        return torch.LongTensor([17])
    elif language == "at":
        return torch.LongTensor([18])
    elif language == "vd":
        return torch.LongTensor([19])
    elif language == "at-lab":
        return torch.LongTensor([20])


if __name__ == '__main__':
    #tf = ArticulatoryCombinedTextFrontend(language="en")
    #tf.string_to_tensor("This is a complex sentence, it even has a pause! But can it do this? Nice.", view=True)

    tf = ArticulatoryCombinedTextFrontend(language="at-lab")
    tf.string_to_tensor("Hi( - Alles klar, jetzt. testen wir einen deutschen Satz... Ich hoffe.. es gibt nicht mehr viele unspezifizierte Phoneme. Unter uns, fuhr!!! fuhr?", view=True, path_to_wavfile="/data/vokquant/data/aridialect/aridialect_wav16000/alf_at_berlin_001.wav")
    
    #tf = ArticulatoryCombinedTextFrontend(language="at")
    #tf.string_to_tensor("Hi( - Alles klar, jetzt. testen wir einen deutschen Satz... Ich hoffe.. es gibt nicht mehr viele unspezifizierte Phoneme. Unter uns, fuhr!!! fuhr?", view=True, path_to_wavfile="")

    #tf = ArticulatoryCombinedTextFrontend(language="cmn")
    #tf.string_to_tensor("这是一个复杂的句子，它甚至包含一个停顿。", view=True)
    #tf.string_to_tensor("李绅 《悯农》    锄禾日当午，    汗滴禾下土。    谁知盘中餐，    粒粒皆辛苦。", view=True)
    #tf.string_to_tensor("巴	拔	把	爸	吧", view=True)

    #tf = ArticulatoryCombinedTextFrontend(language="vi")
    #tf.string_to_tensor("Xin chào thế giới, quả là một ngày tốt lành để học nói tiếng Việt!", view=True)
    #tf.string_to_tensor("ba bà bá bạ bả bã", view=True)
