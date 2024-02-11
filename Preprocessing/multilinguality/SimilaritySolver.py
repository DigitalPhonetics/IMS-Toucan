import json
import os

from Preprocessing.multilinguality.create_map_and_tree_dist_lookups import CacheCreator


class SimilaritySolver:
    def __init__(self):
        self.lang_1_to_lang_2_to_tree_dist = load_json_from_path('lang_1_to_lang_2_to_tree_dist.json')
        self.lang_1_to_lang_2_to_map_dist = load_json_from_path('lang_1_to_lang_2_to_map_dist.json')
        self.iso_to_fullname = load_json_from_path("iso_to_fullname.json")
        pop_keys = list()
        for el in self.iso_to_fullname:
            if "Sign Language" in self.iso_to_fullname[el]:
                pop_keys.append(el)
        for pop_key in pop_keys:
            self.iso_to_fullname.pop(pop_key)
        with open('iso_to_fullname.json', 'w', encoding='utf-8') as f:
            json.dump(self.iso_to_fullname, f, ensure_ascii=False, indent=4)

    def find_closest_in_family(self, lang, supervised_langs, n_closest=5, verbose=False):
        langs_to_sim = dict()
        for supervised_lang in supervised_langs:
            try:
                langs_to_sim[supervised_lang] = self.lang_1_to_lang_2_to_tree_dist[lang][supervised_lang]
            except KeyError:
                try:
                    langs_to_sim[supervised_lang] = self.lang_1_to_lang_2_to_tree_dist[supervised_lang][lang]
                except KeyError:
                    pass
        results = sorted(langs_to_sim, key=langs_to_sim.get, reverse=False)[:n_closest]
        if verbose:
            print(f"\n{n_closest} most similar languages to {self.iso_to_fullname[lang]} according to the Phylogenetic Language Tree from the given list are:")
            for result in results:
                try:
                    print(self.iso_to_fullname[result])
                except KeyError:
                    print("Full Name of Language Missing")
            if len(results) == 0:
                print(f"No matches found for {self.iso_to_fullname[lang]}")
        return results

    def find_closest_on_map(self, lang, n_closest=5, verbose=False):
        langs_to_dist = dict()
        for lang_1 in self.lang_1_to_lang_2_to_map_dist:
            for lang_2 in self.lang_1_to_lang_2_to_map_dist[lang_1]:
                if not lang_1 == lang_2:
                    if lang == lang_1:
                        langs_to_dist[lang_2] = self.lang_1_to_lang_2_to_map_dist[lang_1][lang_2]
                    elif lang == lang_2:
                        langs_to_dist[lang_1] = self.lang_1_to_lang_2_to_map_dist[lang_1][lang_2]
        results = sorted(langs_to_dist, key=langs_to_dist.get, reverse=False)[:n_closest]
        if verbose:
            print(f"\n{n_closest} closest languages to {self.iso_to_fullname[lang]} on the worldmap are:")
            for result in results:
                try:
                    print(self.iso_to_fullname[result])
                except KeyError:
                    print("Full Name of Language Missing")
        return results


def load_json_from_path(path):
    with open(path, "r", encoding="utf8") as f:
        obj = json.loads(f.read())

    return obj


if __name__ == '__main__':
    if not (os.path.exists("lang_1_to_lang_2_to_map_dist.json") and os.path.exists(
            "lang_1_to_lang_2_to_tree_dist.json")):
        CacheCreator()

    ss = SimilaritySolver()

    ss.find_closest_in_family(lang="swg",
                              supervised_langs=['eng', 'deu', 'fra', 'spa', 'cmn', 'por', 'pol', 'ita', 'nld', 'ell', 'fin', 'vie', 'rus', 'hun', 'bem', 'swh', 'amh', 'wol', 'mal', 'chv', 'iba', 'jav', 'fon', 'hau', 'lbb', 'kik', 'lin', 'lug', 'luo', 'sxb', 'yor', 'nya', 'loz', 'toi', 'afr', 'arb', 'asm', 'ast', 'azj', 'bel', 'bul', 'ben', 'bos', 'cat',
                                                'ceb', 'sdh', 'ces', 'cym', 'dan', 'ekk', 'pes', 'fil', 'gle', 'glg', 'guj', 'heb', 'hin', 'hrv', 'hye', 'ind', 'ibo', 'isl', 'kat', 'kam', 'kea', 'kaz', 'khm', 'kan', 'kor', 'ltz', 'lao', 'lit', 'lvs', 'mri', 'mkd', 'xng', 'mar', 'zsm', 'mlt', 'oci', 'ory', 'pan', 'pst', 'ron', 'snd', 'slk', 'slv', 'sna',
                                                'som', 'srp', 'swe', 'tam', 'tel', 'tgk', 'tur', 'ukr', 'umb', 'urd', 'uzn', 'bhd', 'kfs', 'dgo', 'gbk', 'bgc', 'xnr', 'kfx', 'mjl', 'bfz', 'acf', 'bss', 'inb', 'nca', 'quh', 'wap', 'acr', 'bus', 'dgr', 'maz', 'nch', 'qul', 'tav', 'wmw', 'acu', 'byr', 'dik', 'iou', 'mbb', 'ncj', 'qvc', 'tbc', 'xed', 'agd',
                                                'bzh', 'djk', 'ipi', 'mbc', 'ncl', 'qve', 'tbg', 'xon', 'agg', 'bzj', 'dop', 'jac', 'mbh', 'ncu', 'qvh', 'tbl', 'xtd', 'agn', 'caa', 'jic', 'mbj', 'ndj', 'qvm', 'tbz', 'xtm', 'agr', 'cab', 'emp', 'jiv', 'mbt', 'nfa', 'qvn', 'tca', 'yaa', 'agu', 'cap', 'jvn', 'mca', 'ngp', 'qvs', 'tcs', 'yad', 'aia', 'car',
                                                'ese', 'mcb', 'ngu', 'qvw', 'yal', 'cax', 'kaq', 'mcd', 'nhe', 'qvz', 'tee', 'ycn', 'ake', 'cbc', 'far', 'mco', 'qwh', 'yka', 'alp', 'cbi', 'kdc', 'mcp', 'nhu', 'qxh', 'ame', 'cbr', 'gai', 'kde', 'mcq', 'nhw', 'qxn', 'tew', 'yre', 'amf', 'cbs', 'gam', 'kdl', 'mdy', 'nhy', 'qxo', 'tfr', 'yva', 'amk', 'cbt',
                                                'geb', 'kek', 'med', 'nin', 'rai', 'zaa', 'apb', 'cbu', 'glk', 'ken', 'mee', 'nko', 'rgu', 'zab', 'apr', 'cbv', 'meq', 'tgo', 'zac', 'arl', 'cco', 'gng', 'kje', 'met', 'nlg', 'rop', 'tgp', 'zad', 'grc', 'klv', 'mgh', 'nnq', 'rro', 'zai', 'ata', 'cek', 'gub', 'kmu', 'mib', 'noa', 'ruf', 'tna', 'zam', 'atb',
                                                'cgc', 'guh', 'kne', 'mie', 'not', 'rug', 'tnk', 'zao', 'atg', 'chf', 'knf', 'mih', 'npl', 'tnn', 'zar', 'awb', 'chz', 'gum', 'knj', 'mil', 'sab', 'tnp', 'zas', 'cjo', 'guo', 'ksr', 'mio', 'obo', 'seh', 'toc', 'zav', 'azg', 'cle', 'gux', 'kue', 'mit', 'omw', 'sey', 'tos', 'zaw', 'azz', 'cme', 'gvc', 'kvn',
                                                'miz', 'ood', 'sgb', 'tpi', 'zca', 'bao', 'cni', 'gwi', 'kwd', 'mkl', 'shp', 'tpt', 'zga', 'bba', 'cnl', 'gym', 'kwf', 'mkn', 'ote', 'sja', 'trc', 'ziw', 'bbb', 'cnt', 'gyr', 'kwi', 'mop', 'otq', 'snn', 'ttc', 'zlm', 'cof', 'hat', 'kyc', 'mox', 'pab', 'snp', 'tte', 'zos', 'bgt', 'con', 'kyf', 'mpm', 'pad',
                                                'tue', 'zpc', 'bjr', 'cot', 'kyg', 'mpp', 'soy', 'tuf', 'zpl', 'bjv', 'cpa', 'kyq', 'mpx', 'pao', 'tuo', 'zpm', 'bjz', 'cpb', 'hlt', 'kyz', 'mqb', 'pib', 'spp', 'zpo', 'bkd', 'cpu', 'hns', 'lac', 'mqj', 'pir', 'spy', 'txq', 'zpu', 'blz', 'crn', 'hto', 'lat', 'msy', 'pjt', 'sri', 'txu', 'zpz', 'bmr', 'cso',
                                                'hub', 'lex', 'mto', 'pls', 'srm', 'udu', 'ztq', 'bmu', 'ctu', 'lgl', 'muy', 'poi', 'srn', 'zty', 'bnp', 'cuc', 'lid', 'mxb', 'stp', 'upv', 'zyp', 'boa', 'cui', 'huu', 'mxq', 'sus', 'ura', 'boj', 'cuk', 'huv', 'llg', 'mxt', 'poy', 'suz', 'urb', 'box', 'cwe', 'hvn', 'prf', 'urt', 'bpr', 'cya', 'ign', 'lww',
                                                'myk', 'ptu', 'usp', 'bps', 'daa', 'ikk', 'maj', 'myy', 'vid', 'bqc', 'dah', 'nab', 'qub', 'tac', 'bqp', 'ded', 'imo', 'maq', 'nas', 'quf', 'taj', 'vmy'],
                              n_closest=5,
                              verbose=True)

    ss.find_closest_on_map(lang="swg",
                           n_closest=10,
                           verbose=True)
