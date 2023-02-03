import json
import os.path

from tqdm import tqdm


class CacheCreator:
    def __init__(self):
        self.iso_codes = list(load_json_from_path("iso_to_fullname.json").keys())

        self.pairs = list()  # ignore order, collect all language pairs
        for index_1 in tqdm(range(len(self.iso_codes))):
            for index_2 in range(index_1, len(self.iso_codes)):
                self.pairs.append((self.iso_codes[index_1], self.iso_codes[index_2]))

        iso_to_familiy_memberships = load_json_from_path("iso_to_memberships.json")

        #######
        self.pair_to_tree_dist = dict()
        for pair in tqdm(self.pairs):
            self.pair_to_tree_dist[pair] = len(
                set(iso_to_familiy_memberships[pair[0]]).intersection(set(iso_to_familiy_memberships[pair[1]])))
        pruning_keys = list()
        for key in tqdm(self.pair_to_tree_dist):
            if self.pair_to_tree_dist[key] < 2:
                pruning_keys.append(key)
        for key in pruning_keys:
            self.pair_to_tree_dist.pop(key)
        # approx 2mio pairs with a tree similarity of 2 or higher left
        lang_1_to_lang_2_to_tree_dist = dict()
        for pair in self.pair_to_tree_dist:
            lang_1 = pair[0]
            lang_2 = pair[1]
            dist = self.pair_to_tree_dist[pair]
            if lang_1 not in lang_1_to_lang_2_to_tree_dist.keys():
                lang_1_to_lang_2_to_tree_dist[lang_1] = dict()
            lang_1_to_lang_2_to_tree_dist[lang_1][lang_2] = dist
        with open('Preprocessing/lang_1_to_lang_2_to_tree_dist.json', 'w', encoding='utf-8') as f:
            json.dump(lang_1_to_lang_2_to_tree_dist, f, ensure_ascii=False, indent=4)

        #######
        self.pair_to_map_dist = dict()
        iso_to_long_lat = load_json_from_path("iso_to_long_lat.json")
        for pair in tqdm(self.pairs):
            try:
                long, lat = iso_to_long_lat[pair[0]]
                long_2, lat_2 = iso_to_long_lat[pair[1]]
                self.pair_to_map_dist[pair] = abs(((long + 9999) - (long_2 + 9999)) + ((lat + 9999) - (lat_2 + 9999)))
            except KeyError:
                pass
        lang_1_to_lang_2_to_map_dist = dict()
        for pair in self.pair_to_map_dist:
            lang_1 = pair[0]
            lang_2 = pair[1]
            dist = self.pair_to_map_dist[pair]
            if lang_1 not in lang_1_to_lang_2_to_map_dist.keys():
                lang_1_to_lang_2_to_map_dist[lang_1] = dict()
            lang_1_to_lang_2_to_map_dist[lang_1][lang_2] = dist

        with open('Preprocessing/lang_1_to_lang_2_to_map_dist.json', 'w', encoding='utf-8') as f:
            json.dump(lang_1_to_lang_2_to_map_dist, f, ensure_ascii=False, indent=4)

    def find_closest_in_family(self, lang, supervised_langs, n_closest=5):
        langs_to_sim = dict()
        for supervised_lang in supervised_langs:
            try:
                langs_to_sim[supervised_lang] = self.pair_to_tree_dist[(lang, supervised_lang)]
            except KeyError:
                try:
                    langs_to_sim[supervised_lang] = self.pair_to_tree_dist[(supervised_lang, lang)]
                except KeyError:
                    pass
        return sorted(langs_to_sim, key=langs_to_sim.get, reverse=True)[:n_closest]

    def find_closest_on_map(self, lang, n_closest=5):
        langs_to_dist = dict()
        for pair in self.pair_to_map_dist:
            if lang in pair:
                if lang == pair[0]:
                    langs_to_dist[pair[1]] = self.pair_to_map_dist[pair]
                else:
                    langs_to_dist[pair[0]] = self.pair_to_map_dist[pair]
        return sorted(langs_to_dist, key=langs_to_dist.get, reverse=False)[:n_closest]


def load_json_from_path(path):
    with open(path, "r", encoding="utf8") as f:
        obj = json.loads(f.read())
    return obj


if __name__ == '__main__':
    if not (os.path.exists("lang_1_to_lang_2_to_map_dist.json") and os.path.exists(
            "lang_1_to_lang_2_to_tree_dist.json")):
        CacheCreator()
