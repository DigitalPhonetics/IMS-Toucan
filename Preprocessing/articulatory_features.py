# partly derived from an open-source resource provided by Papercup Technologies Limited
# Resource-Author: Marlene Staib
# Modified by Florian Lux, 2021
# Further modified by Florian Lux, 2022

def generate_feature_lookup():
    return {
        '~': {'symbol_type': 'silence'},
        '#': {'symbol_type': 'end of sentence'},
        '?': {'symbol_type': 'questionmark'},
        '!': {'symbol_type': 'exclamationmark'},
        '.': {'symbol_type': 'fullstop'},
        'ɜ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'vowel',
            'VUV': 'voiced',
            'vowel_frontness': 'central',
            'vowel_openness': 'open-mid',
            'vowel_roundedness': 'unrounded',
        },
        'ɫ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'alveolar',
            'consonant_manner': 'lateral-approximant',
        },
        'ə': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'vowel',
            'VUV': 'voiced',
            'vowel_frontness': 'central',
            'vowel_openness': 'mid',
            'vowel_roundedness': 'unrounded',
        },
        'ɚ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'vowel',
            'VUV': 'voiced',
            'vowel_frontness': 'central',
            'vowel_openness': 'mid',
            'vowel_roundedness': 'unrounded',
        },
        'a': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'vowel',
            'VUV': 'voiced',
            'vowel_frontness': 'front',
            'vowel_openness': 'open',
            'vowel_roundedness': 'unrounded',
        },
        'ð': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'dental',
            'consonant_manner': 'fricative'
        },
        'ɛ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'vowel',
            'VUV': 'voiced',
            'vowel_frontness': 'front',
            'vowel_openness': 'open-mid',
            'vowel_roundedness': 'unrounded',
        },
        'ɪ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'vowel',
            'VUV': 'voiced',
            'vowel_frontness': 'front_central',
            'vowel_openness': 'close_close-mid',
            'vowel_roundedness': 'unrounded',
        },
        'ᵻ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'vowel',
            'VUV': 'voiced',
            'vowel_frontness': 'central',
            'vowel_openness': 'close',
            'vowel_roundedness': 'unrounded',
        },
        'ŋ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'velar',
            'consonant_manner': 'nasal'
        },
        'ɔ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'vowel',
            'VUV': 'voiced',
            'vowel_frontness': 'back',
            'vowel_openness': 'open-mid',
            'vowel_roundedness': 'rounded',
        },
        'ɒ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'vowel',
            'VUV': 'voiced',
            'vowel_frontness': 'back',
            'vowel_openness': 'open',
            'vowel_roundedness': 'rounded',
        },
        'ɾ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'alveolar',
            'consonant_manner': 'flap'
        },
        'ʃ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'unvoiced',
            'consonant_place': 'postalveolar',
            'consonant_manner': 'fricative'
        },
        'θ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'unvoiced',
            'consonant_place': 'dental',
            'consonant_manner': 'fricative'
        },
        'ʊ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'vowel',
            'VUV': 'voiced',
            'vowel_frontness': 'central_back',
            'vowel_openness': 'close_close-mid',
            'vowel_roundedness': 'unrounded'
        },
        'ʌ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'vowel',
            'VUV': 'voiced',
            'vowel_frontness': 'back',
            'vowel_openness': 'open-mid',
            'vowel_roundedness': 'unrounded'
        },
        'ʒ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'postalveolar',
            'consonant_manner': 'fricative'
        },
        'æ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'vowel',
            'VUV': 'voiced',
            'vowel_frontness': 'front',
            'vowel_openness': 'open-mid_open',
            'vowel_roundedness': 'unrounded'
        },
        'b': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'bilabial',
            'consonant_manner': 'plosive'
        },
        'ʔ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'unvoiced',
            'consonant_place': 'glottal',
            'consonant_manner': 'plosive'
        },
        'd': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'alveolar',
            'consonant_manner': 'plosive'
        },
        'e': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'vowel',
            'VUV': 'voiced',
            'vowel_frontness': 'front',
            'vowel_openness': 'close-mid',
            'vowel_roundedness': 'unrounded'
        },
        'f': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'unvoiced',
            'consonant_place': 'labiodental',
            'consonant_manner': 'fricative'
        },
        'g': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'velar',
            'consonant_manner': 'plosive'
        },
        'h': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'unvoiced',
            'consonant_place': 'glottal',
            'consonant_manner': 'fricative'
        },
        'i': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'vowel',
            'VUV': 'voiced',
            'vowel_frontness': 'front',
            'vowel_openness': 'close',
            'vowel_roundedness': 'unrounded'
        },
        'j': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'palatal',
            'consonant_manner': 'approximant'
        },
        'k': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'unvoiced',
            'consonant_place': 'velar',
            'consonant_manner': 'plosive'
        },
        'l': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'alveolar',
            'consonant_manner': 'lateral-approximant'
        },
        'm': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'bilabial',
            'consonant_manner': 'nasal'
        },
        'n': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'alveolar',
            'consonant_manner': 'nasal'
        },
        'ɳ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'palatal',
            'consonant_manner': 'nasal'
        },
        'o': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'vowel',
            'VUV': 'voiced',
            'vowel_frontness': 'back',
            'vowel_openness': 'close-mid',
            'vowel_roundedness': 'rounded'
        },
        'p': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'unvoiced',
            'consonant_place': 'bilabial',
            'consonant_manner': 'plosive'
        },
        'ɡ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'velar',
            'consonant_manner': 'plosive'
        },
        'ɹ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'alveolar',
            'consonant_manner': 'approximant'
        },
        'r': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'alveolar',
            'consonant_manner': 'trill'
        },
        's': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'unvoiced',
            'consonant_place': 'alveolar',
            'consonant_manner': 'fricative'
        },
        't': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'unvoiced',
            'consonant_place': 'alveolar',
            'consonant_manner': 'plosive'
        },
        'u': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'vowel',
            'VUV': 'voiced',
            'vowel_frontness': 'back',
            'vowel_openness': 'close',
            'vowel_roundedness': 'rounded',
        },
        'v': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'labiodental',
            'consonant_manner': 'fricative'
        },
        'w': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'labial-velar',
            'consonant_manner': 'approximant'
        },
        'x': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'unvoiced',
            'consonant_place': 'velar',
            'consonant_manner': 'fricative'
        },
        'z': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'alveolar',
            'consonant_manner': 'fricative'
        },
        'ʀ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'uvular',
            'consonant_manner': 'trill'
        },
        'ø': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'vowel',
            'VUV': 'voiced',
            'vowel_frontness': 'front',
            'vowel_openness': 'close-mid',
            'vowel_roundedness': 'rounded'
        },
        'ç': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'unvoiced',
            'consonant_place': 'palatal',
            'consonant_manner': 'fricative'
        },
        'ɐ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'vowel',
            'VUV': 'voiced',
            'vowel_frontness': 'central',
            'vowel_openness': 'open',
            'vowel_roundedness': 'unrounded'
        },
        'œ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'vowel',
            'VUV': 'voiced',
            'vowel_frontness': 'front',
            'vowel_openness': 'open-mid',
            'vowel_roundedness': 'rounded'
        },
        'y': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'vowel',
            'VUV': 'voiced',
            'vowel_frontness': 'front',
            'vowel_openness': 'close',
            'vowel_roundedness': 'rounded'
        },
        'ʏ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'vowel',
            'VUV': 'voiced',
            'vowel_frontness': 'front_central',
            'vowel_openness': 'close_close-mid',
            'vowel_roundedness': 'rounded'
        },
        'ɑ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'vowel',
            'VUV': 'voiced',
            'vowel_frontness': 'back',
            'vowel_openness': 'open',
            'vowel_roundedness': 'unrounded'
        },
        'c': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'unvoiced',
            'consonant_place': 'palatal',
            'consonant_manner': 'plosive'
        },
        'ɲ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'palatal',
            'consonant_manner': 'nasal'
        },
        'ɣ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'velar',
            'consonant_manner': 'fricative'
        },
        'ʎ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'palatal',
            'consonant_manner': 'lateral-approximant'
        },
        'β': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'bilabial',
            'consonant_manner': 'fricative'
        },
        'ʝ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'palatal',
            'consonant_manner': 'fricative'
        },
        'ɟ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'palatal',
            'consonant_manner': 'plosive'
        },
        'q': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'unvoiced',
            'consonant_place': 'uvular',
            'consonant_manner': 'plosive'
        },
        'ɕ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'unvoiced',
            'consonant_place': 'alveolopalatal',
            'consonant_manner': 'fricative'
        },
        'ɭ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'retroflex',
            'consonant_manner': 'lateral-approximant'
        },
        'ɵ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'vowel',
            'VUV': 'voiced',
            'vowel_frontness': 'central',
            'vowel_openness': 'open-mid',
            'vowel_roundedness': 'rounded'
        },
        'ʑ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'alveolopalatal',
            'consonant_manner': 'fricative'
        },
        'ʋ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'labiodental',
            'consonant_manner': 'approximant'
        },
        'ʁ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'uvular',
            'consonant_manner': 'fricative'
        },
        'ɨ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'vowel',
            'VUV': 'voiced',
            'vowel_frontness': 'central',
            'vowel_openness': 'close',
            'vowel_roundedness': 'unrounded'
        },
        'ʂ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'unvoiced',
            'consonant_place': 'retroflex',
            'consonant_manner': 'fricative'
        },
        'ɬ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'unvoiced',
            'consonant_place': 'alveolar',
            'consonant_manner': 'fricative'  # it's a lateral fricative, but should be close enough
        },
        'ɓ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'bilabial',
            'consonant_manner': 'implosive'
        },
        'ʙ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'bilabial',
            'consonant_manner': 'vibrant'
        },
        'ɗ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'dental',
            'consonant_manner': 'implosive'
        },
        'ɖ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'retroflex',
            'consonant_manner': 'plosive'
        },
        'χ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'unvoiced',
            'consonant_place': 'uvular',
            'consonant_manner': 'fricative'
        },
        'ʛ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'uvular',
            'consonant_manner': 'implosive'
        },
        'ʟ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'velar',
            'consonant_manner': 'lateral-approximant'
        },
        'ɮ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'alveolar',
            'consonant_manner': 'fricative'
        },
        'ɽ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'retroflex',
            'consonant_manner': 'flap'
        },
        'ɢ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'uvular',
            'consonant_manner': 'plosive'
        },
        'ɠ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'velar',
            'consonant_manner': 'implosive'
        },
        'ǂ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'unvoiced',
            'consonant_place': 'alveolopalatal',
            'consonant_manner': 'click'
        },
        'ɦ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'glottal',
            'consonant_manner': 'fricative'
        },
        'ǁ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'unvoiced',
            'consonant_place': 'alveolar',
            'consonant_manner': 'click'
        },
        'ĩ': {  # identical description with i except nasal
            'symbol_type': 'phoneme',
            'vowel_consonant': 'vowel',
            'VUV': 'voiced',
            'vowel_frontness': 'front',
            'vowel_openness': 'close',
            'vowel_roundedness': 'unrounded',
            'consonant_manner': 'nasal'
        },
        'ʍ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'unvoiced',
            'consonant_place': 'labial-velar',
            'consonant_manner': 'fricative'
        },
        'ʕ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'pharyngal',
            'consonant_manner': 'fricative'
        },
        'ɻ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'unvoiced',
            'consonant_place': 'retroflex',
            'consonant_manner': 'approximant'
        },
        'ʄ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'palatal',
            'consonant_manner': 'implosive'
        },
        'ɺ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'alveolar',
            'consonant_manner': 'flap'  # it's a lateral flap, but should be close enough
        },
        'ũ': {  # identical with u, but nasal
            'symbol_type': 'phoneme',
            'vowel_consonant': 'vowel',
            'VUV': 'voiced',
            'vowel_frontness': 'back',
            'vowel_openness': 'close',
            'vowel_roundedness': 'rounded',
            'consonant_manner': 'nasal'
        },
        'ɤ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'vowel',
            'VUV': 'voiced',
            'vowel_frontness': 'back',
            'vowel_openness': 'close-mid',
            'vowel_roundedness': 'unrounded',
        },
        'ɶ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'vowel',
            'VUV': 'voiced',
            'vowel_frontness': 'front',
            'vowel_openness': 'open',
            'vowel_roundedness': 'rounded',
        },
        'õ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'vowel',
            'VUV': 'voiced',
            'vowel_frontness': 'back',
            'vowel_openness': 'close-mid',
            'vowel_roundedness': 'rounded',
            'consonant_manner': 'nasal'
        },

    }  # REMEMBER to also add the phonemes added here to the ID lookup table below as the new highest ID


def get_phone_to_id():
    """
    for the states of the ctc loss and dijkstra/mas in the aligner
    cannot be extracted trivially from above because sets are unordered and the IDs need to be consistent
    """
    return {
        '~': 0,
        '#': 1,
        '?': 2,
        '!': 3,
        '.': 4,
        'ɜ': 5,
        'ɫ': 6,
        'ə': 7,
        'ɚ': 8,
        'a': 9,
        'ð': 10,
        'ɛ': 11,
        'ɪ': 12,
        'ᵻ': 13,
        'ŋ': 14,
        'ɔ': 15,
        'ɒ': 16,
        'ɾ': 17,
        'ʃ': 18,
        'θ': 19,
        'ʊ': 20,
        'ʌ': 21,
        'ʒ': 22,
        'æ': 23,
        'b': 24,
        'ʔ': 25,
        'd': 26,
        'e': 27,
        'f': 28,
        'g': 29,
        'h': 30,
        'i': 31,
        'j': 32,
        'k': 33,
        'l': 34,
        'm': 35,
        'n': 36,
        'ɳ': 37,
        'o': 38,
        'p': 39,
        'ɡ': 40,
        'ɹ': 41,
        'r': 42,
        's': 43,
        't': 44,
        'u': 45,
        'v': 46,
        'w': 47,
        'x': 48,
        'z': 49,
        'ʀ': 50,
        'ø': 51,
        'ç': 52,
        'ɐ': 53,
        'œ': 54,
        'y': 55,
        'ʏ': 56,
        'ɑ': 57,
        'c': 58,
        'ɲ': 59,
        'ɣ': 60,
        'ʎ': 61,
        'β': 62,
        'ʝ': 63,
        'ɟ': 64,
        'q': 65,
        'ɕ': 66,
        'ʲ': 67,
        'ɭ': 68,
        'ɵ': 69,
        'ʑ': 70,
        'ʋ': 71,
        'ʁ': 72,
        'ɨ': 73,
        'ʂ': 74,
        'ɬ': 75,
        'ɓ': 76,
        'ʙ': 77,
        'ɗ': 78,
        'ɖ': 79,
        'χ': 80,
        'ʛ': 81,
        'ʟ': 82,
        'ɽ': 83,
        'ɢ': 84,
        'ɠ': 85,
        'ǂ': 86,
        'ɦ': 87,
        'ǁ': 88,
        'ĩ': 89,
        'ʍ': 90,
        'ʕ': 91,
        'ɻ': 92,
        'ʄ': 93,
        'ɺ': 94,
        'ũ': 95,
        'ɤ': 96,
        'ɶ': 97,
        'õ': 98,
    }


# ʡ
# ʈ
# ʜ
# ɱ
# ɯ
# ǀ
# ɧ
# ɸ
# ʘ
# ʐ
# ɰ
# ɘ
# ɥ
# ħ
# ɞ
# ʉ
# ɴ
# ʢ
# ѵ


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

    value_to_index = {
        "dental": 0,
        "postalveolar": 1,
        "mid": 2,
        "close-mid": 3,
        "vowel": 4,
        "silence": 5,
        "consonant": 6,
        "close": 7,
        "velar": 8,
        "plosive": 9,
        "palatal": 10,
        "nasal": 11,
        "glottal": 12,
        "central": 13,
        "back": 14,
        "approximant": 15,
        "uvular": 16,
        "open-mid": 17,
        "front_central": 18,
        "front": 19,
        "end of sentence": 20,
        "labiodental": 21,
        "close_close-mid": 22,
        "labial-velar": 23,
        "unvoiced": 24,
        "central_back": 25,
        "trill": 26,
        "rounded": 27,
        "open-mid_open": 28,
        "flap": 29,
        "alveolar": 30,
        "bilabial": 31,
        "phoneme": 32,
        "open": 33,
        "fricative": 34,
        "unrounded": 35,
        "lateral-approximant": 36,
        "voiced": 37,
        "questionmark": 38,
        "exclamationmark": 39,
        "fullstop": 40,
        "alveolopalatal": 41,
        "lengthened": 42,  # modified by following symbol
        "half-length": 43,  # modified by following symbol
        "shortened": 44,  # modified by following symbol
        "stressed": 45,  # modified by previous symbol
        "very-high-tone": 46,  # modified by following symbol
        "high-tone": 47,  # modified by following symbol
        "mid-tone": 48,  # modified by following symbol
        "low-tone": 49,  # modified by following symbol
        "very-low-tone": 50,  # modified by following symbol
        "implosive": 51,
        "vibrant": 52,
        "retroflex": 53,
        "click": 54,
        "pharyngal": 55,
    }

    phone_to_vector = dict()
    for ipa in ipa_to_phonemefeats:
        if len(ipa) == 1:
            phone_to_vector[ipa] = [0] * sum([len(values) for values in [feat_to_val_set[feat] for feat in feat_to_val_set]])
            for feat in ipa_to_phonemefeats[ipa]:
                if ipa_to_phonemefeats[ipa][feat] in value_to_index:
                    phone_to_vector[ipa][value_to_index[ipa_to_phonemefeats[ipa][feat]]] = 1

    for feat in feat_to_val_set:
        for value in feat_to_val_set[feat]:
            if value not in value_to_index:
                print(f"Unknown feature value in featureset! {value}")

    # print(f"{sum([len(values) for values in [feat_to_val_set[feat] for feat in feat_to_val_set]])} should be 42")

    return phone_to_vector

# postalveolar click is not supported because the IPA symbol overlaps with the excalamation mark, which I find to be more important in most languages.
