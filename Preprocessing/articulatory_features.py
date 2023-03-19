# -*- coding: utf-8 -*-


# partly derived from an open-source resource provided by Papercup Technologies Limited
# Resource-Author: Marlene Staib
# Modified by Florian Lux, 2021
# Further modified by Florian Lux, 2022


"""
All phonemes in the IPA standard are supported.

zero-width characters are generally not supported, as
well as some other modifiers. Tone, stress and
lengthening are represented with placeholder dimensions,
however they need to be set manually, this conversion
from phonemes to features works on a character by
character basis. In a few cases, the place of
articulation is approximated because only one phoneme
had such a combination, which does not warrant a new
dimension.
"""


def generate_feature_lookup():
    return {
        '~': {'symbol_type': 'silence'},
        '#': {'symbol_type': 'end of sentence'},
        '?': {'symbol_type': 'questionmark'},
        '!': {'symbol_type': 'exclamationmark'},
        '.': {'symbol_type': 'fullstop'},
        ' ': {'symbol_type': 'word-boundary'},
        'ɜ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'central',
            'vowel_openness'   : 'open-mid',
            'vowel_roundedness': 'unrounded',
        },
        'ə': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'central',
            'vowel_openness'   : 'mid',
            'vowel_roundedness': 'unrounded',
        },
        'a': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'front',
            'vowel_openness'   : 'open',
            'vowel_roundedness': 'unrounded',
        },
        'ð': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'dental',
            'consonant_manner': 'fricative'
        },
        'ɛ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'front',
            'vowel_openness'   : 'open-mid',
            'vowel_roundedness': 'unrounded',
        },
        'ɪ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'front_central',
            'vowel_openness'   : 'close_close-mid',
            'vowel_roundedness': 'unrounded',
        },
        'ŋ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'velar',
            'consonant_manner': 'nasal'
        },
        'ɔ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'back',
            'vowel_openness'   : 'open-mid',
            'vowel_roundedness': 'rounded',
        },
        'ɒ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'back',
            'vowel_openness'   : 'open',
            'vowel_roundedness': 'rounded',
        },
        'ɾ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'alveolar',
            'consonant_manner': 'flap'
        },
        'ʃ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'postalveolar',
            'consonant_manner': 'fricative'
        },
        'θ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'dental',
            'consonant_manner': 'fricative'
        },
        'ʊ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'central_back',
            'vowel_openness'   : 'close_close-mid',
            'vowel_roundedness': 'unrounded'
        },
        'ʌ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'back',
            'vowel_openness'   : 'open-mid',
            'vowel_roundedness': 'unrounded'
        },
        'ʒ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'postalveolar',
            'consonant_manner': 'fricative'
        },
        'æ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'front',
            'vowel_openness'   : 'open-mid_open',
            'vowel_roundedness': 'unrounded'
        },
        'b': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'bilabial',
            'consonant_manner': 'plosive'
        },
        'ʔ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'glottal',
            'consonant_manner': 'plosive'
        },
        'd': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'alveolar',
            'consonant_manner': 'plosive'
        },
        'e': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'front',
            'vowel_openness'   : 'close-mid',
            'vowel_roundedness': 'unrounded'
        },
        'f': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'labiodental',
            'consonant_manner': 'fricative'
        },
        'ɡ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'velar',
            'consonant_manner': 'plosive'
        },
        'h': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'glottal',
            'consonant_manner': 'fricative'
        },
        'i': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'front',
            'vowel_openness'   : 'close',
            'vowel_roundedness': 'unrounded'
        },
        'j': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'palatal',
            'consonant_manner': 'approximant'
        },
        'k': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'velar',
            'consonant_manner': 'plosive'
        },
        'l': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'alveolar',
            'consonant_manner': 'lateral-approximant'
        },
        'm': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'bilabial',
            'consonant_manner': 'nasal'
        },
        'n': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'alveolar',
            'consonant_manner': 'nasal'
        },
        'ɳ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'retroflex',
            'consonant_manner': 'nasal'
        },
        'o': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'back',
            'vowel_openness'   : 'close-mid',
            'vowel_roundedness': 'rounded'
        },
        'p': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'bilabial',
            'consonant_manner': 'plosive'
        },
        'ɹ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'alveolar',
            'consonant_manner': 'approximant'
        },
        'r': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'alveolar',
            'consonant_manner': 'trill'
        },
        's': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'alveolar',
            'consonant_manner': 'fricative'
        },
        't': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'alveolar',
            'consonant_manner': 'plosive'
        },
        'u': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'back',
            'vowel_openness'   : 'close',
            'vowel_roundedness': 'rounded',
        },
        'v': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'labiodental',
            'consonant_manner': 'fricative'
        },
        'w': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'labial-velar',
            'consonant_manner': 'approximant'
        },
        'x': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'velar',
            'consonant_manner': 'fricative'
        },
        'z': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'alveolar',
            'consonant_manner': 'fricative'
        },
        'ʀ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'uvular',
            'consonant_manner': 'trill'
        },
        'ø': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'front',
            'vowel_openness'   : 'close-mid',
            'vowel_roundedness': 'rounded'
        },
        'ç': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'palatal',
            'consonant_manner': 'fricative'
        },
        'ɐ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'central',
            'vowel_openness'   : 'open',
            'vowel_roundedness': 'unrounded'
        },
        'œ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'front',
            'vowel_openness'   : 'open-mid',
            'vowel_roundedness': 'rounded'
        },
        'y': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'front',
            'vowel_openness'   : 'close',
            'vowel_roundedness': 'rounded'
        },
        'ʏ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'front_central',
            'vowel_openness'   : 'close_close-mid',
            'vowel_roundedness': 'rounded'
        },
        'ɑ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'back',
            'vowel_openness'   : 'open',
            'vowel_roundedness': 'unrounded'
        },
        'c': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'palatal',
            'consonant_manner': 'plosive'
        },
        'ɲ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'palatal',
            'consonant_manner': 'nasal'
        },
        'ɣ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'velar',
            'consonant_manner': 'fricative'
        },
        'ʎ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'palatal',
            'consonant_manner': 'lateral-approximant'
        },
        'β': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'bilabial',
            'consonant_manner': 'fricative'
        },
        'ʝ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'palatal',
            'consonant_manner': 'fricative'
        },
        'ɟ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'palatal',
            'consonant_manner': 'plosive'
        },
        'q': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'uvular',
            'consonant_manner': 'plosive'
        },
        'ɕ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'alveolopalatal',
            'consonant_manner': 'fricative'
        },
        'ɭ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'retroflex',
            'consonant_manner': 'lateral-approximant'
        },
        'ɵ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'central',
            'vowel_openness'   : 'close-mid',
            'vowel_roundedness': 'rounded'
        },
        'ʑ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'alveolopalatal',
            'consonant_manner': 'fricative'
        },
        'ʋ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'labiodental',
            'consonant_manner': 'approximant'
        },
        'ʁ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'uvular',
            'consonant_manner': 'fricative'
        },
        'ɨ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'central',
            'vowel_openness'   : 'close',
            'vowel_roundedness': 'unrounded'
        },
        'ʂ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'retroflex',
            'consonant_manner': 'fricative'
        },
        'ɓ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'bilabial',
            'consonant_manner': 'implosive'
        },
        'ʙ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'bilabial',
            'consonant_manner': 'vibrant'
        },
        'ɗ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'dental',
            'consonant_manner': 'implosive'
        },
        'ɖ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'retroflex',
            'consonant_manner': 'plosive'
        },
        'χ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'uvular',
            'consonant_manner': 'fricative'
        },
        'ʛ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'uvular',
            'consonant_manner': 'implosive'
        },
        'ʟ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'velar',
            'consonant_manner': 'lateral-approximant'
        },
        'ɽ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'retroflex',
            'consonant_manner': 'flap'
        },
        'ɢ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'uvular',
            'consonant_manner': 'plosive'
        },
        'ɠ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'velar',
            'consonant_manner': 'implosive'
        },
        'ǂ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'alveolopalatal',
            'consonant_manner': 'click'
        },
        'ɦ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'glottal',
            'consonant_manner': 'fricative'
        },
        'ǁ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'alveolar',
            'consonant_manner': 'click'
        },
        'ĩ': {  # identical description with i except nasal
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'front',
            'vowel_openness'   : 'close',
            'vowel_roundedness': 'unrounded',
            'consonant_manner' : 'nasal'
        },
        'ʍ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'labial-velar',
            'consonant_manner': 'fricative'
        },
        'ʕ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'pharyngal',
            'consonant_manner': 'fricative'
        },
        'ɻ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'retroflex',
            'consonant_manner': 'approximant'
        },
        'ʄ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'palatal',
            'consonant_manner': 'implosive'
        },
        'ũ': {  # identical with u, but nasal
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'back',
            'vowel_openness'   : 'close',
            'vowel_roundedness': 'rounded',
            'consonant_manner' : 'nasal'
        },
        'ɤ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'back',
            'vowel_openness'   : 'close-mid',
            'vowel_roundedness': 'unrounded',
        },
        'ɶ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'front',
            'vowel_openness'   : 'open',
            'vowel_roundedness': 'rounded',
        },
        'õ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'back',
            'vowel_openness'   : 'close-mid',
            'vowel_roundedness': 'rounded',
            'consonant_manner' : 'nasal'
        },
        'ʡ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'epiglottal',
            'consonant_manner': 'plosive'
        },
        'ʈ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'retroflex',
            'consonant_manner': 'plosive'
        },
        'ʜ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'epiglottal',
            'consonant_manner': 'fricative'
        },
        'ɱ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'labiodental',
            'consonant_manner': 'nasal'
        },
        'ɯ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'back',
            'vowel_openness'   : 'close',
            'vowel_roundedness': 'unrounded'
        },
        'ǀ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'dental',
            'consonant_manner': 'click'
        },
        'ɸ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'bilabial',
            'consonant_manner': 'fricative'
        },
        'ʘ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'bilabial',
            'consonant_manner': 'click'
        },
        'ʐ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'retroflex',
            'consonant_manner': 'fricative'
        },
        'ɰ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'velar',
            'consonant_manner': 'approximant'
        },
        'ɘ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'central',
            'vowel_openness'   : 'close-mid',
            'vowel_roundedness': 'unrounded'
        },
        'ħ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'pharyngal',
            'consonant_manner': 'fricative'
        },
        'ɞ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'central',
            'vowel_openness'   : 'open-mid',
            'vowel_roundedness': 'rounded'
        },
        'ʉ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'central',
            'vowel_openness'   : 'close',
            'vowel_roundedness': 'rounded'
        },
        'ɴ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'uvular',
            'consonant_manner': 'nasal'
        },
        'ʢ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'epiglottal',
            'consonant_manner': 'fricative'
        },
        'ѵ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'labiodental',
            'consonant_manner': 'flap'
        },
        'ǃ': {  # looks deceivingly like an exclamation mark, but it's a different unicode entry
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'postalveolar',
            'consonant_manner': 'click'
        },

    }  # REMEMBER to also add the phonemes added here to the ID lookup below as the new highest ID


def get_phone_to_id():
    """
    for the states of the ctc loss and dijkstra/mas in the aligner
    cannot be extracted trivially from above because sets are unordered and the IDs need to be consistent
    """
    phone_to_id = dict()
    for index, phone in enumerate("~#?!ǃ.ɜəaðɛɪŋɔɒɾʃθʊʌʒæbʔdefghijklmnɳopɡɹrstuvwxzʀøçɐœyʏɑcɲɣʎβʝɟqɕɭɵʑʋʁɨʂɓʙɗɖχʛʟɽɢɠǂɦǁĩʍʕɻʄũɤɶõʡʈʜɱɯǀɸʘʐɰɘħɞʉɴʢѵ"):
        phone_to_id[phone] = index
    return phone_to_id


def get_feature_to_index_lookup():
    return {
        # MODIFIER
        # -- stress: modified by the previous symbol
        "stressed"           : 0,
        # -- tone: modified by the following symbol
        "very-high-tone"     : 1,
        "high-tone"          : 2,
        "mid-tone"           : 3,
        "low-tone"           : 4,
        "very-low-tone"      : 5,
        "rising-tone"        : 6,
        "falling-tone"       : 7,
        "peaking-tone"       : 8,
        "dipping-tone"       : 9,
        # -- lengthening: modified by the following symbol
        "lengthened"         : 10,
        "half-length"        : 11,
        "shortened"          : 12,

        # CATEGORIES
        "consonant"          : 13,
        "vowel"              : 14,
        "phoneme"            : 15,

        # NON-SPEECH-MARKERS
        "silence"            : 16,
        "end of sentence"    : 17,
        "questionmark"       : 18,
        "exclamationmark"    : 19,
        "fullstop"           : 20,
        "word-boundary"      : 21,

        # PLACE
        "dental"             : 22,
        "postalveolar"       : 23,
        "velar"              : 24,
        "palatal"            : 25,
        "glottal"            : 26,
        "uvular"             : 27,
        "labiodental"        : 28,
        "labial-velar"       : 29,
        "alveolar"           : 30,
        "bilabial"           : 31,
        "alveolopalatal"     : 32,
        "retroflex"          : 33,
        "pharyngal"          : 34,
        "epiglottal"         : 35,

        # TONGUE POSITION
        "central"            : 36,
        "back"               : 37,
        "front_central"      : 38,
        "front"              : 39,
        "central_back"       : 40,

        # MOUTH OPENNESS
        "mid"                : 41,
        "close-mid"          : 42,
        "close"              : 43,
        "open-mid"           : 44,
        "close_close-mid"    : 45,
        "open-mid_open"      : 46,
        "open"               : 47,

        # MOUTH SHAPE
        "rounded"            : 48,
        "unrounded"          : 49,

        # MANNER
        "plosive"            : 50,
        "nasal"              : 51,
        "approximant"        : 52,
        "trill"              : 53,
        "flap"               : 54,
        "fricative"          : 55,
        "lateral-approximant": 56,
        "implosive"          : 57,
        "vibrant"            : 58,
        "click"              : 59,

        # TYPE
        "unvoiced"           : 60,
        "voiced"             : 61,
    }


def generate_feature_table():
    ipa_to_phonemefeats = generate_feature_lookup()

    feat_types = set()
    for ipa in ipa_to_phonemefeats:
        if len(ipa) == 1:
            [feat_types.add(feat) for feat in ipa_to_phonemefeats[ipa].keys()]

    feat_to_val_set = dict()
    for feat in feat_types:
        feat_to_val_set[feat] = set()
    for ipa in ipa_to_phonemefeats:
        if len(ipa) == 1:
            for feat in ipa_to_phonemefeats[ipa]:
                feat_to_val_set[feat].add(ipa_to_phonemefeats[ipa][feat])

    # print(feat_to_val_set)

    value_list = set()
    for val_set in [feat_to_val_set[feat] for feat in feat_to_val_set]:
        for value in val_set:
            value_list.add(value)
    # print("{")
    # for index, value in enumerate(list(value_list)):
    #     print('"{}":{},'.format(value,index))
    # print("}")

    value_to_index = get_feature_to_index_lookup()

    phone_to_vector = dict()
    for ipa in ipa_to_phonemefeats:
        if len(ipa) == 1:
            phone_to_vector[ipa] = [0] * (13 + sum([len(values) for values in [feat_to_val_set[feat] for feat in feat_to_val_set]]))
            # there are 13 features which do not occur in the vectors, because they are context dependent and not lexical
            for feat in ipa_to_phonemefeats[ipa]:
                if ipa_to_phonemefeats[ipa][feat] in value_to_index:
                    phone_to_vector[ipa][value_to_index[ipa_to_phonemefeats[ipa][feat]]] = 1

    for feat in feat_to_val_set:
        for value in feat_to_val_set[feat]:
            if value not in value_to_index:
                print(f"Unknown feature value in featureset! {value}")

    # print(f"{sum([len(values) for values in [feat_to_val_set[feat] for feat in feat_to_val_set]])} should be 49")

    return phone_to_vector


if __name__ == '__main__':
    print(generate_feature_table())
