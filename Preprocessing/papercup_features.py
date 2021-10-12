# Derived from  an open-source resource provided by Papercup Technologies Limited
# Resource-Author: Marlene Staib
# Modified by Florian Lux, 2021

def generate_feature_table():
    ipa_to_phonemefeats = {
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
            'consonant_manner': 'tap'
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
            'consonant_manner': 'stop'
        },
        'ʔ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'unvoiced',
            'consonant_place': 'glottal',
            'consonant_manner': 'stop'
        },
        'd': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'alveolar',
            'consonant_manner': 'stop'
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
            'consonant_manner': 'stop'
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
            'consonant_manner': 'stop'
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
            'consonant_manner': 'stop'
        },
        'ɡ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'velar',
            'consonant_manner': 'stop'
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
            'consonant_manner': 'stop'
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
            'consonant_manner': 'stop'
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
            'consonant_manner': 'stop'
        },
        'q': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'unvoiced',
            'consonant_place': 'uvular',
            'consonant_manner': 'stop'
        },
        'ɕ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'unvoiced',
            'consonant_place': 'alveolopalatal',
            'consonant_manner': 'fricative'
        },
        'ʲ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'palatal',
            'consonant_manner': 'approximant'
        },
        'ɭ': {
            'symbol_type': 'phoneme',
            'vowel_consonant': 'consonant',
            'VUV': 'voiced',
            'consonant_place': 'palatal',  # should be retroflex, but palatal should be close enough
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
    }

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
        "stop": 9,
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
        "tap": 29,
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
        "alveolopalatal": 41
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
