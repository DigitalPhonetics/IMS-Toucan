import argparse
import os
import pickle
from copy import deepcopy

import pandas as pd
from tqdm import tqdm

from Preprocessing.multilinguality.SimilaritySolver import SimilaritySolver
from Utility.storage_config import MODELS_DIR
from Utility.utils import load_json_from_path

ISO_LOOKUP_PATH = "iso_lookup.json"
ISO_TO_FULLNAME_PATH = "iso_to_fullname.json"
LANG_PAIRS_MAP_PATH = "lang_1_to_lang_2_to_map_dist.json"
LANG_PAIRS_TREE_PATH = "lang_1_to_lang_2_to_tree_dist.json"
LANG_PAIRS_ASP_PATH = "asp_dict.pkl"
LANG_PAIRS_LEARNED_DIST_PATH = "lang_1_to_lang_2_to_learned_dist.json"
LANG_PAIRS_ORACLE_PATH = "lang_1_to_lang_2_to_oracle_dist.json"
SUPVERVISED_LANGUAGES_PATH = "supervised_languages.json"
DATASET_SAVE_DIR = "distance_datasets/"


class LangDistDatasetCreator():
    def __init__(self, model_path, cache_root="."):
        self.model_path = model_path
        self.cache_root = cache_root
        self.lang_pairs_map = None
        self.largest_value_map_dist = None
        self.lang_pairs_tree = None
        self.lang_pairs_asp = None
        self.lang_pairs_learned_dist = None
        self.lang_pairs_oracle = None
        self.supervised_langs = load_json_from_path(os.path.join(cache_root, SUPVERVISED_LANGUAGES_PATH))
        self.iso_lookup = load_json_from_path(os.path.join(cache_root, ISO_LOOKUP_PATH))
        self.iso_to_fullname = load_json_from_path(os.path.join(cache_root, ISO_TO_FULLNAME_PATH))

    def load_required_distance_lookups(self, distance_type, excluded_distances=[]):
        # init required distance lookups
        print(f"Loading required distance lookups for distance_type '{distance_type}'.")
        try:
            if distance_type == "combined":
                if "map" not in excluded_distances and not self.lang_pairs_map:
                    self.lang_pairs_map = load_json_from_path(os.path.join(self.cache_root, LANG_PAIRS_MAP_PATH))
                    self.largest_value_map_dist = 0.0
                    for _, values in self.lang_pairs_map.items():
                        for _, value in values.items():
                            self.largest_value_map_dist = max(self.largest_value_map_dist, value)
                if "tree" not in excluded_distances and not self.lang_pairs_tree:
                    self.lang_pairs_tree = load_json_from_path(os.path.join(self.cache_root, LANG_PAIRS_TREE_PATH))
                if "asp" not in excluded_distances and not self.lang_pairs_asp:
                    with open(os.path.join(self.cache_root, LANG_PAIRS_ASP_PATH), "rb") as f:
                        self.lang_pairs_asp = pickle.load(f)
            elif distance_type == "map" and not self.lang_pairs_map:
                self.lang_pairs_map = load_json_from_path(os.path.join(self.cache_root, LANG_PAIRS_MAP_PATH))
                self.largest_value_map_dist = 0.0
                for _, values in self.lang_pairs_map.items():
                    for _, value in values.items():
                        self.largest_value_map_dist = max(self.largest_value_map_dist, value)
            elif distance_type == "tree" and not self.lang_pairs_tree:
                self.lang_pairs_tree = load_json_from_path(os.path.join(self.cache_root, LANG_PAIRS_TREE_PATH))
            elif distance_type == "asp" and not self.lang_pairs_asp:
                with open(os.path.join(self.cache_root, LANG_PAIRS_ASP_PATH), "rb") as f:
                    self.lang_pairs_asp = pickle.load(f)
            elif distance_type == "learned" and not self.lang_pairs_learned_dist:
                self.lang_pairs_learned_dist = load_json_from_path(os.path.join(self.cache_root, LANG_PAIRS_LEARNED_DIST_PATH))
            elif distance_type == "oracle" and not self.lang_pairs_oracle:
                self.lang_pairs_oracle = load_json_from_path(os.path.join(self.cache_root, LANG_PAIRS_ORACLE_PATH))
        except FileNotFoundError as e:
            raise FileNotFoundError("Please create all lookup files via create_distance_lookups.py") from e

    def create_dataset(self,
                       distance_type: str = "learned",
                       zero_shot: bool = False,
                       n_closest: int = 50,
                       excluded_languages: list = [],
                       excluded_distances: list = [],
                       find_furthest: bool = False,
                       individual_distances: bool = False,
                       write_to_csv=True):
        """Create dataset with a given feature's distance in a dict, and saves it to a CSV file."""
        distance_types = ["learned", "map", "tree", "asp", "combined", "random", "oracle"]
        if distance_type not in distance_types:
            raise ValueError(f"Invalid distance type '{distance_type}'. Expected one of {distance_types}")
        dataset_dict = dict()
        self.load_required_distance_lookups(distance_type, excluded_distances)

        sim_solver = SimilaritySolver(tree_dist=self.lang_pairs_tree,
                                      map_dist=self.lang_pairs_map,
                                      largest_value_map_dist=self.largest_value_map_dist,
                                      asp_dict=self.lang_pairs_asp,
                                      learned_dist=self.lang_pairs_learned_dist,
                                      oracle_dist=self.lang_pairs_oracle,
                                      iso_to_fullname=self.iso_to_fullname)
        supervised_langs = sorted(self.supervised_langs)
        remove_langs_suffix = ""
        if len(excluded_languages) > 0:
            remove_langs_suffix = "_no-illegal-langs"
            for excl_lang in excluded_languages:
                supervised_langs.remove(excl_lang)
        individual_dist_suffix, excluded_feat_suffix = "", ""
        if distance_type == "combined":
            if individual_distances:
                individual_dist_suffix = "_indiv-dists"
            if len(excluded_distances) > 0:
                excluded_feat_suffix = "_excl-" + "-".join(excluded_distances)
        furthest_suffix = "_furthest" if find_furthest else ""
        zero_shot_suffix = ""
        if zero_shot:
            iso_codes_to_ids = deepcopy(self.iso_lookup)[-1]
            zero_shot_suffix = "_zeroshot"
            # leave supervised-pretrained language embeddings untouched
            for sup_lang in supervised_langs:
                iso_codes_to_ids.pop(sup_lang, None)
            lang_codes = list(iso_codes_to_ids)
        else:
            lang_codes = supervised_langs
        failed_langs = []
        if distance_type == "random":
            random_seed = 0
        sorted_by = "closest" if not find_furthest else "furthest"

        for lang in tqdm(lang_codes, desc=f"Retrieving {sorted_by} distances"):
            if distance_type == "combined":
                feature_dict = sim_solver.find_closest_combined_distance(lang,
                                                                         supervised_langs,
                                                                         k=n_closest,
                                                                         individual_distances=individual_distances,
                                                                         excluded_features=excluded_distances,
                                                                         find_furthest=find_furthest)
            elif distance_type == "random":
                random_seed += 1
                dataset_dict[lang] = [lang]  # target language as first column
                feature_dict = sim_solver.find_closest(distance_type,
                                                       lang,
                                                       supervised_langs,
                                                       k=n_closest,
                                                       find_furthest=find_furthest,
                                                       random_seed=random_seed)
            else:
                feature_dict = sim_solver.find_closest(distance_type,
                                                       lang,
                                                       supervised_langs,
                                                       k=n_closest,
                                                       find_furthest=find_furthest)
            # discard incomplete results
            if len(feature_dict) < n_closest:
                failed_langs.append(lang)
                continue

            dataset_dict[lang] = [lang]  # target language as first column
            # create entry for a single close lang (`feature_dict` must be sorted by distance)
            for _, close_lang in enumerate(feature_dict):
                if distance_type == "combined":
                    dist_combined = feature_dict[close_lang]["combined_distance"]
                    close_lang_feature_list = [close_lang, dist_combined]
                    if individual_distances:
                        indiv_dists = feature_dict[close_lang]["individual_distances"]
                        close_lang_feature_list.extend(indiv_dists)
                else:
                    dist = feature_dict[close_lang]
                    close_lang_feature_list = [close_lang, dist]
                # column order: compared close language, {feature}_dist (plus optionally indiv dists)
                dataset_dict[lang].extend(close_lang_feature_list)

        # prepare df columns
        dataset_columns = ["target_lang"]
        for i in range(n_closest):
            dataset_columns.extend([f"closest_lang_{i}", f"{distance_type}_dist_{i}"])
            if distance_type == "combined" and individual_distances:
                if "map" not in excluded_distances:
                    dataset_columns.append(f"map_dist_{i}")
                if "asp" not in excluded_distances:
                    dataset_columns.append(f"asp_dist_{i}")
                if "tree" not in excluded_distances:
                    dataset_columns.append(f"tree_dist_{i}")
        df = pd.DataFrame.from_dict(dataset_dict, orient="index")
        df.columns = dataset_columns

        if write_to_csv:
            out_path = os.path.join(os.path.join(self.cache_root, DATASET_SAVE_DIR), f"dataset_{distance_type}_top{n_closest}{furthest_suffix}{zero_shot_suffix}{remove_langs_suffix}{excluded_feat_suffix}{individual_dist_suffix}" + ".csv")
            os.makedirs(os.path.join(self.cache_root, DATASET_SAVE_DIR), exist_ok=True)
            df.to_csv(out_path, sep="|", index=False)
        print(f"Successfully retrieved distances for {len(lang_codes) - len(failed_langs)}/{len(lang_codes)} languages.")
        if len(failed_langs) > 0:
            print(f"Failed to retrieve distances for the following {len(failed_langs)} languages:\n{failed_langs}")
        return df


if __name__ == "__main__":
    default_model_path = os.path.join(MODELS_DIR, "ToucanTTS_Meta", "best.pt")  # MODELS_DIR must be absolute path, the relative path will fail at this location
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-m", type=str, default=default_model_path, help="model path from which to obtain pretrained language embeddings")
    parser.add_argument("--learned_dist_path", type=str, default="lang_1_to_lang_2_to_learned_dist.json",
                        help="filepath of JSON file containing the meta-learned pairwise distances")
    args = parser.parse_args()

    dc = LangDistDatasetCreator(args.model_path)

    excluded_langs = []

    # create datasets for evaluation of approx. lang emb methods on supervised languages
    dataset = dc.create_dataset(distance_type="tree", n_closest=30, zero_shot=False)
    dataset = dc.create_dataset(distance_type="map", n_closest=30, zero_shot=False, excluded_languages=excluded_langs)
    dataset = dc.create_dataset(distance_type="map", n_closest=30, zero_shot=False, find_furthest=True)
    dataset = dc.create_dataset(distance_type="asp", n_closest=30, zero_shot=False)
    dataset = dc.create_dataset(distance_type="random", n_closest=30, zero_shot=False, excluded_languages=excluded_langs)
    dataset = dc.create_dataset(distance_type="combined", n_closest=30, zero_shot=False, individual_distances=True)
    dataset = dc.create_dataset(distance_type="learned", n_closest=30, zero_shot=False)
    dataset = dc.create_dataset(distance_type="oracle", n_closest=30, zero_shot=False)
