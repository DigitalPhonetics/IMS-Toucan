import json
import pickle
import os
import numpy as np
import random
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
        self.largest_value_map_dist = 0.0
        for _, values in self.lang_1_to_lang_2_to_map_dist.items():
            for _, value in values.items():
                self.largest_value_map_dist = max(self.largest_value_map_dist, value)
        if asp_dict:
            self.asp_dict = asp_dict
        else:
            asp_dict_path = "asp_dict.pkl" if not asp_dict_path else asp_dict_path
            with open(asp_dict_path, "rb") as f:
                self.asp_dict = pickle.load(f)
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
        langs_to_dist = dict()
        supervised_langs = set(supervised_langs) if isinstance(supervised_langs, list) else supervised_langs
        if "urk" in supervised_langs:
            supervised_langs.remove("urk")        
        if lang in supervised_langs:
            supervised_langs.remove(lang)
        for sup_lang in supervised_langs:
            dist = self.get_tree_distance(lang, sup_lang)
            if dist is not None:
                langs_to_dist[sup_lang] = dist
        results = dict(sorted(langs_to_dist.items(), key=lambda x: x[1], reverse=False)[:n_closest])
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
        if "urk" in supervised_langs:
            supervised_langs.remove("urk")
        if lang in supervised_langs:
            supervised_langs.remove(lang)
        for sup_lang in supervised_langs:
            dist = self.get_map_distance(lang, sup_lang)
            if dist is not None:
                langs_to_dist[sup_lang] = dist
        results = dict(sorted(langs_to_dist.items(), key=lambda x: x[1], reverse=False)[:n_closest])
        if verbose:
            print(f"{n_closest} closest languages to {self.iso_to_fullname[lang]} in the given list on the worldmap are:")
            for result in results:
                try:
                    print(self.iso_to_fullname[result])
                except KeyError:
                    print("Full Name of Language Missing")
        return results


    def get_random_languages(self, lang, supervised_langs, n=5, random_seed=42):
        """Get n random languages that should be treated as `closest` to the target language.
        The similarity/distance value is always 0.5."""
        supervised_langs = set(supervised_langs) if isinstance(supervised_langs, list) else supervised_langs
        if "urk" in supervised_langs:
            supervised_langs.remove("urk")
        if lang in supervised_langs:
            supervised_langs.remove(lang)
        random.seed(random_seed)
        random_langs = random.sample(supervised_langs, n)
        # create dict with all 0.5 values
        random_dict = {rand_lang: 0.5 for rand_lang in random_langs}
        return random_dict

    def find_closest_aspf(self, lang, supervised_langs, n_closest=5, verbose=False):
        """Find the closest n languages in terms of Angular Similarity of Phoneme Frequencies (ASPF)"""
        langs_to_sim = dict()
        supervised_langs = set(supervised_langs) if isinstance(supervised_langs, list) else supervised_langs
        if "urk" in supervised_langs:
            supervised_langs.remove("urk")
        if lang in supervised_langs:
            supervised_langs.remove(lang)
        for supervised_lang in supervised_langs:
            asp_score = asp(lang, supervised_lang, self.asp_dict)
            if asp_score is not None:
                langs_to_sim[supervised_lang] = asp_score
        results = dict(sorted(langs_to_sim.items(), key=lambda x: x[1], reverse=True)[:n_closest])
        if verbose:
            print(f"{n_closest} closest languages to {self.iso_to_fullname[lang]} w.r.t. ASPF are:")
            for result in results:
                try:
                    print(self.iso_to_fullname[result])
                    print(results[result])
                except KeyError:
                    print("Full Name of Language Missing")
        return results
    

    def find_closest_combined(self, 
                              lang, 
                              supervised_langs, 
                              distance, 
                              n_closest=5, 
                              individual_distances=False, 
                              verbose=False, 
                              learned_weights=False):
        """Find the n closest languages according to the combined Euclidean distance of map distance, tree distance, and ASP distance.
        Returns a dict of dicts of the format {"supervised_lang_1": 
                                                {"euclidean_distance": 5.39, "individual_distances": [<map_dist>, <tree_dist>, <asp_dist>]},
                                              "supervised_lang_2":
                                                {...}, ...}"""
        if distance not in ["average", "euclidean"]:
            raise ValueError
        combined_dict = {}
        supervised_langs = set(supervised_langs) if isinstance(supervised_langs, list) else supervised_langs
        if "urk" in supervised_langs:
            supervised_langs.remove("urk")
        if lang in supervised_langs:
            supervised_langs.remove(lang)
        for sup_lang in supervised_langs:
            map_dist = self.get_map_distance(lang, sup_lang)
            tree_dist = self.get_tree_distance(lang, sup_lang)
            asp_score = asp(lang, sup_lang, self.asp_dict)
            # if getting one of the scores failed, ignore this language
            if None in {map_dist, tree_dist, asp_score}:
                continue           
            
            combined_dict[sup_lang] = {}
            asp_dist = 1 - asp_score # turn into dist since other 2 are also dists
            if learned_weights:
                dist_array = np.array([0.0128*map_dist, 0.4611*tree_dist, 0.3058*asp_dist]) # apply learned weights
            else:
                dist_array = np.array([map_dist, tree_dist, asp_dist])
            #map_sim = 1 - map_dist # turn into sim since other 2 are also sims
            #dist_array = np.array([map_sim, tree_dist, asp_score])
            if distance == "euclidean":
                euclidean_dist = np.sqrt(np.sum(dist_array**2)) # no subtraction because lang has dist [0,0,0]
                combined_dict[sup_lang]["combined_distance"] = euclidean_dist
                #combined_dict[sup_lang] = {"combined_distance": euclidean_dist, "individual_distances": [map_sim, tree_dist, asp_score]}
            elif distance == "average":
                avg_dist = np.mean(dist_array)
                combined_dict[sup_lang]["combined_distance"] = avg_dist
            else:
                raise ValueError("distance needs to be `average` or `euclidean`")                
            if individual_distances:
                combined_dict[sup_lang]["individual_distances"] = [map_dist, tree_dist, asp_dist]
                #combined_dict[sup_lang] = {"combined_distance": avg_dist, "individual_distances": [map_sim, tree_dist, asp_score]}

        # results = dict(sorted(combined_dict.items(), key=lambda x: x[1]["combined_distance"])[:n_closest])
        results = dict(sorted(combined_dict.items(), key=lambda x: x[1]["combined_distance"], reverse=False)[:n_closest])
        if verbose:
            print(f"{n_closest} closest languages to {self.iso_to_fullname[lang]} w.r.t. the combined features are:")
            for result in results:
                try:
                    print(self.iso_to_fullname[result])
                    print(results[result])
                except KeyError:
                    print("Full Name of Language Missing")
        return results

    def get_map_distance(self, lang_1, lang_2):
        """Returns normalized map distance between two languages.
        If no value can be retrieved, returns None."""
        try:
            dist = self.lang_1_to_lang_2_to_map_dist[lang_1][lang_2]
        except KeyError:
            try:
                dist = self.lang_1_to_lang_2_to_map_dist[lang_2][lang_1]
            except KeyError:
                return None
        dist = dist / self.largest_value_map_dist # normalize
        return dist
    
    def get_tree_distance(self, lang_1, lang_2):
        """Returns normalized tree distance between two languages.
        If no value can be retrieved, returns None."""
        try:
            dist = self.lang_1_to_lang_2_to_tree_dist[lang_1][lang_2]
        except KeyError:
            try:
                dist = self.lang_1_to_lang_2_to_tree_dist[lang_2][lang_1]
            except KeyError:
                return None
        return dist

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
    
    ss.find_closest_combined("vie", ["dga",
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
                                      "dhl"], 
                                      distance="average", 
                                      n_closest=5, 
                                      verbose=True)

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
