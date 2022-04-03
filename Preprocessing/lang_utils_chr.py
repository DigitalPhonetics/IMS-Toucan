import re
import unicodedata as un

from typing import Dict


# '\u02e5': 76,  # ◌˥
# '\u02e6': 77,  # ◌˦
# '\u02e7': 78,  # ◌˧
# '\u02e8': 79,  # ◌˨
# '\u02e9': 80,  # ◌˩

def chr_mco_ipa(mco_text: str) -> str:
    if not mco_text:
        return ""
    lookup = get_mco_lookup_dict()
    ipa_text = ""
    # decompose combined diacritics for simplified conversion to IPA tone letters
    mco_text = mco_text.lower()
    mco_text = un.normalize("NFD", mco_text)
    mco_text = re.sub("\\s+", " ", mco_text)  # normalize spacing
    mco_text = re.sub(":([^\\s])", "\u02d0\\1", mco_text)  # convert intra-word colons into IPA long vowel letters
    mco_text = mco_text.replace("ch", "tʃ")  # a one-off multi-char convert
    for ch in mco_text:
        if ch in lookup:
            ipa_text += lookup[ch]
        else:
            ipa_text += ch
    # The vowel length letter needs to precede the tone letters.
    ipa_text = re.sub("([\u02e5-\u02e9]+)(\u02d0)", "\\2\\1", ipa_text)
    return ipa_text


def get_mco_lookup_dict() -> Dict[str, str]:
    lookup: Dict[str, str] = dict()
    lookup["\u0304"] = "\u02e8"  # ˨ low tone, combining macro
    lookup["\u0300"] = "\u02e8\u02e9"  # ˨˩ low fall tone, combining grave accent
    lookup["\u030c"] = "\u02e8\u02e7"  # ˨˧ rising tone, combining caron
    lookup["\u0302"] = "\u02e7\u02e8"  # ˧˨ falling tone, combining circumflex accent
    lookup["\u0301"] = "\u02e7"  # ˧ high tone, combining acute accent
    lookup["\u030b"] = "\u02e7\u02e6"  # ˧˦ superhigh (high rising tone), combining double acute accent
    lookup["ɂ"] = "ʔ"
    lookup["v"] = "ə̃"
    lookup["j"] = "dʒ"
    lookup["y"] = "j"
    return lookup


def test():
    print(chr_mco_ipa("""
Anǐ:táɂli ani:sgaya à:ni:no:halǐ:dô:he, ahwi dù:ni:hyohe.
Sà:gwű:hno asgaya galò:gwé ga:ne:he sóɂíhnv́ hlā.
Ná:hnv́ galò:gwé ga:ne̋:hi u:dlv̌:kwsati gè:sé, ale go:hű:sdi yǔ:dv̂:ne̋:la à:dlv̌:kwsgé.
À:ná:ɂi:sv̋:hnv go:hű:sdi wǔ:ní:go:he do:juwáɂihlv, ná:hnv́ galò:gwé ga:ne̋:hi kilagwu iyv̋:da widǔ:sdáyo:hlé ǒ:sdagwu nǔ:ksestanv̋:na iyú:sdi dà:sdayo:hihv̋.
U:do:hiyű:hnv́ wǔ:yó:hlé ale ù:ni:go:hé ganv́:gv̋.
Ná:hnv́ galò:gwé nigǎ:né:hv̋:na "Ahwi è:ni:yó:ɂa!", ù:dv:hne.
"Ji:yó:ɂê:ga", ù:dv:hne ná galò:gwé ga:ne̋:hi, à:dlv̌:kwsgv́.
Ù:ná:ne:lǔ:gî:se do:juwáɂihlv́ dí:dla, naɂv̌:hníge̋:hnv wǔ:ní:luhja ù:ni:go:hé sǒ:gwíli gáɂnv̋.
"Sǒ:gwílílê ì:nada:hísi", ù:dv:hné ná u:yo:hlv̋.
"Hada:hísê:gá", à:gò:sě:lé.
"""))


if __name__ == '__main__':
    test()
