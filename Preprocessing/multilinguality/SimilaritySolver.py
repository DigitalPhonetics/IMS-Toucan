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
            print(f"{n_closest} closest languages to {self.iso_to_fullname[lang]} on the worldmap are:")
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
                                      "dkx"], 5, True)

    ss.find_closest_on_map("vie", 10, True)
