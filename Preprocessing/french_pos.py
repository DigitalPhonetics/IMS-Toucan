def map_to_wiktionary_pos():  
    return {
        #conjonction
        'COSUB': 'conjonction',
        'COCO': 'conjonction',
        'CCONJ': 'conjonction',
        'SCONJ': 'conjonction',

        # pronom personnel
        'PPER1S': 'pronom personnel',
        'PPER2S': 'pronom personnel',
        'PPER3MS': 'pronom personnel',
        'PPER3FS': 'pronom personnel',
        'PPER3MP': 'pronom personnel',
        'PPER3FP': 'pronom personnel',
        'PPOBJMS': 'pronom personnel',
        'PPOBJFS': 'pronom personnel',
        'PPOBJMP': 'pronom personnel',
        'PPOBJFP': 'pronom personnel',

        # adjectif
        'ADJ': 'adjectif',
        'ADJMS': 'adjectif',
        'ADJFS': 'adjectif',
        'ADJMP': 'adjectif',
        'ADJFP': 'adjectif',

        # interjection
        'INTJ': 'interjection',

        # nom
        'NOUN': 'nom',
        'NMS': 'nom',
        'NFS': 'nom',
        'NMP': 'nom',
        'NFP': 'nom',

        # det
        'DET': 'det',
        'DETMS': 'det',
        'DETFS': 'det',

        # pronom indéfini
        'PINDMS': 'pronom indéfini',
        'PINDFS': 'pronom indéfini',
        'PINDMP': 'pronom indéfini',
        'PINDFP': 'pronom indéfini',

        # nom propre
        'PROPN': 'nom propre',

        # nom de famille
        'XFAMIL': 'nom de famille',

        # adjectif numéral
        'NUM': 'adjectif numéral',
        'DINTMS': 'adjectif',
        'DINTFS': 'adjectif',

        # onomatopée -> ignore this, there is only one instance in homograph list

        # préposition
        'PREP': 'préposition',
        'ADP': 'préposition',

        # adverbe
        'ADV': 'adverbe',

        # verb
        'AUX': 'verbe',
        'VERB': 'verbe',
        'VPPMS': 'verbe',
        'VPPFS': 'verbe',
        'VPPMP': 'verbe',
        'VPPFP': 'verbe'

        # TODO: what about PRON? This can be any kind of pronoun and ambigiuities occur with personal pronouns and indefinite pronouns

        # Taggs we don't have to care about:
        # O, PDEMMS, YPFOR, PUNCT, PREL, PREF, CHIF, PRELFS, SYM, MOTINC, PREFP, PDEMFS PART, X 
    }