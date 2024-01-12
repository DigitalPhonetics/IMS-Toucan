import json
import pickle
import os
# TODO: remove sys.path.append
import sys
sys.path.append("/home/behringe/hdd_behringe/IMS-Toucan")

from Preprocessing.multilinguality.create_map_and_tree_dist_lookups import CacheCreator
from Preprocessing.multilinguality.asp import asp, load_asp_dict


class SimilaritySolver():
    def __init__(self, 
                 tree_dist=None, 
                 map_dist=None, 
                 asp_dict=None,
                 tree_dist_path=None, 
                 map_dist_path=None, 
                 asp_dict_path=None,
                 iso_to_fullname=None, 
                 iso_to_fullname_path=None):
        if tree_dist:
            self.lang_1_to_lang_2_to_tree_dist = tree_dist
        else:
            tree_dist_path = 'lang_1_to_lang_2_to_tree_dist.json' if not tree_dist_path else tree_dist_path
            self.lang_1_to_lang_2_to_tree_dist = load_json_from_path(tree_dist_path)
        if map_dist:
            self.lang_1_to_lang_2_to_map_dist = map_dist
        else:
            map_dist_path = 'lang_1_to_lang_2_to_map_dist.json' if not map_dist_path else map_dist_path
            self.lang_1_to_lang_2_to_map_dist = load_json_from_path(map_dist_path)
        if asp_dict:
            self.asp_dict = asp_dict
        else:
            asp_dict_path = "asp_dict.pkl" if not asp_dict_path else asp_dict_path
            with open(asp_dict_path, "rb") as f:
                self.asp_dict = pickle.load(asp_dict_path)
        if iso_to_fullname:
            self.iso_to_fullname = iso_to_fullname
        else:
            iso_to_fullname_path = "iso_to_fullname.json" if not iso_to_fullname_path else iso_to_fullname_path
        self.iso_to_fullname = load_json_from_path(iso_to_fullname_path)
        
        pop_keys = list()
        for el in self.iso_to_fullname:
            if "Sign Language" in self.iso_to_fullname[el]:
                pop_keys.append(el)
        for pop_key in pop_keys:
            self.iso_to_fullname.pop(pop_key)
        with open(iso_to_fullname_path, 'w', encoding='utf-8') as f:
            json.dump(self.iso_to_fullname, f, ensure_ascii=False, indent=4)

    def find_closest_in_family(self, lang, supervised_langs, n_closest=5, verbose=False):
        langs_to_sim = dict()
        supervised_langs = set(supervised_langs) if isinstance(supervised_langs, list) else supervised_langs
        if lang in supervised_langs:
            supervised_langs.remove(lang)
        for supervised_lang in supervised_langs:
            try:
                langs_to_sim[supervised_lang] = self.lang_1_to_lang_2_to_tree_dist[lang][supervised_lang]
            except KeyError:
                try:
                    langs_to_sim[supervised_lang] = self.lang_1_to_lang_2_to_tree_dist[supervised_lang][lang]
                except KeyError:
                    pass
        results = sorted(langs_to_sim, key=langs_to_sim.get, reverse=True)[:n_closest]
        if verbose:
            print(f"{n_closest} most similar languages to {self.iso_to_fullname[lang]} in the given list are:")
            for result in results:
                try:
                    print(self.iso_to_fullname[result])
                except KeyError:
                    print("Full Name of Language Missing")
            if len(results) == 0:
                print(f"No matches found for {self.iso_to_fullname[lang]}")
        return results

    def find_closest_on_map(self, lang, supervised_langs, n_closest=5, verbose=False):
        """Find the closest n supervised languages on the map, i.e. for which language embeddings are available."""
        langs_to_dist = dict()
        supervised_langs = set(supervised_langs) if isinstance(supervised_langs, list) else supervised_langs
        if lang in supervised_langs:
            supervised_langs.remove(lang)
        for supervised_lang in supervised_langs:
            try:
                langs_to_dist[supervised_lang] = self.lang_1_to_lang_2_to_map_dist[lang][supervised_lang]
            except KeyError:
                try:
                    langs_to_dist[supervised_lang] = self.lang_1_to_lang_2_to_map_dist[supervised_lang][lang]
                except KeyError:
                    pass
        results = sorted(langs_to_dist, key=langs_to_dist.get, reverse=False)[:n_closest]
        if verbose:
            print(f"{n_closest} closest languages to {self.iso_to_fullname[lang]} in the given list on the worldmap are:")
            for result in results:
                try:
                    print(self.iso_to_fullname[result])
                except KeyError:
                    print("Full Name of Language Missing")
        return results


    def find_closest_aspf(self, lang, supervised_langs, n_closest=5, verbose=False):
        """Find the closest n languages in terms of Angular Similarity of Phoneme Frequencies (ASPF)"""
        langs_to_sim = dict()
        supervised_langs = set(supervised_langs) if isinstance(supervised_langs, list) else supervised_langs
        if lang in supervised_langs:
            supervised_langs.remove(lang)
        for supervised_lang in supervised_langs:
            try:
                langs_to_sim[supervised_lang] = asp(lang, supervised_lang, self.asp_dict)
            except KeyError:
                pass
        results = sorted(langs_to_sim, key=langs_to_sim.get, reverse=True)[:n_closest]
        if verbose:
            print(f"{n_closest} closest languages to {self.iso_to_fullname[lang]} w.r.t. ASPF are:")
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

    ss.find_closest_aspf("vie", ["dga",
                                      "dgb",
                                      "dgc",
                                      "dgd",
                                      "dge",
                                      "dgg",
                                      "dgh",
                                      "dgi",
                                      "dgk",
                                      "dgl",
                                      "dgn",
                                      "dgo",
                                      "dgr",
                                      "dgs",
                                      "dgt",
                                      "dgw",
                                      "dgx",
                                      "dgz",
                                      "dhd",
                                      "dhg",
                                      "dhi",
                                      "dhl"], 5, verbose=True)
    
    ss.find_closest_in_family("vie", ["dga",
                                      "dgb",
                                      "dgc",
                                      "dgd",
                                      "dge",
                                      "dgg",
                                      "dgh",
                                      "dgi",
                                      "dgk",
                                      "dgl",
                                      "dgn",
                                      "dgo",
                                      "dgr",
                                      "dgs",
                                      "dgt",
                                      "dgw",
                                      "dgx",
                                      "dgz",
                                      "dhd",
                                      "dhg",
                                      "dhi",
                                      "dhl",
                                      "dhm",
                                      "dhn",
                                      "dho",
                                      "dhr",
                                      "dhs",
                                      "dhu",
                                      "dhv",
                                      "dhw",
                                      "dhx",
                                      "dia",
                                      "dib",
                                      "dic",
                                      "did",
                                      "dif",
                                      "dig",
                                      "dih",
                                      "dii",
                                      "dij",
                                      "dik",
                                      "dil",
                                      "dim",
                                      "din",
                                      "dio",
                                      "dip",
                                      "diq",
                                      "dir",
                                      "dis",
                                      "dit",
                                      "diu",
                                      "div",
                                      "diw",
                                      "dix",
                                      "diy",
                                      "diz",
                                      "djb",
                                      "djc",
                                      "djd",
                                      "dje",
                                      "djf",
                                      "dji",
                                      "djj",
                                      "djk",
                                      "djl",
                                      "djm",
                                      "djn",
                                      "djo",
                                      "djr",
                                      "dju",
                                      "djw",
                                      "dka",
                                      "dkk",
                                      "dkr",
                                      "dks",
                                      "dkx",
                                      "vie"], 5, True)

    ss.find_closest_on_map("vie", ["dga",
                                      "dgb",
                                      "dgc",
                                      "dgd",
                                      "dge",
                                      "dgg",
                                      "dgh",
                                      "dgi",
                                      "dgk",
                                      "dgl",
                                      "dgn",
                                      "dgo",
                                      "dgr",
                                      "dgs",
                                      "dgt",
                                      "dgw",
                                      "dgx",
                                      "dgz",
                                      "dhd",
                                      "dhg",
                                      "dhi",
                                      "dhl",
                                      "dhm",
                                      "dhn",
                                      "dho",
                                      "dhr",
                                      "dhs",
                                      "dhu",
                                      "dhv",
                                      "dhw",
                                      "dhx",
                                      "dia",
                                      "dib",
                                      "dic",
                                      "did",
                                      "dif",
                                      "dig",
                                      "dih",
                                      "dii",
                                      "dij",
                                      "dik",
                                      "dil",
                                      "dim",
                                      "din",
                                      "dio",
                                      "dip",
                                      "diq",
                                      "dir",
                                      "dis",
                                      "dit",
                                      "diu",
                                      "div",
                                      "diw",
                                      "dix",
                                      "diy",
                                      "diz",
                                      "djb",
                                      "djc",
                                      "djd",
                                      "dje",
                                      "djf",
                                      "dji",
                                      "djj",
                                      "djk",
                                      "djl",
                                      "djm",
                                      "djn",
                                      "djo",
                                      "djr",
                                      "dju",
                                      "djw",
                                      "dka",
                                      "dkk",
                                      "dkr",
                                      "dks",
                                      "dkx",
                                      "vie"], 10, True)
