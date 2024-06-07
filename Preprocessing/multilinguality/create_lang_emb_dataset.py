import numpy as np
import pandas as pd
import os
import random
import json
import yaml
import pickle
import torch
from tqdm import tqdm
from copy import deepcopy

from Preprocessing.TextFrontend import load_json_from_path
from Preprocessing.multilinguality.SimilaritySolver import SimilaritySolver
import argparse

ISO_LOOKUP_PATH = "iso_lookup.json"
ISO_TO_FULLNAME_PATH = "iso_to_fullname.json"
LANG_PAIRS_MAP_PATH = "lang_1_to_lang_2_to_map_dist.json"
LANG_PAIRS_TREE_PATH = "lang_1_to_lang_2_to_tree_dist.json"
LANG_PAIRS_ASP_PATH = "asp_dict.pkl"
LANG_PAIRS_LEARNED_DIST_PATH = "lang_1_to_lang_2_to_learned_dist_focal4.json"
NUM_LANGS = 463
LOSS_TYPE = "with_less_loss_fixed_tree_distance"
LANG_EMBS_PATH = f"LangEmbs/final_model_{LOSS_TYPE}.pt"

LANG_EMBS_MAPPING_PATH = f"LangEmbs/mapping_lang_embs_{NUM_LANGS}_langs.yaml"
# TODO: get lang_embs in a nicer way than from this mapping

DATASET_SAVE_DIR = "new_datasets/"


class LangDistDatasetCreator():
    def __init__(self, learned_dist_path=None):
        (self.lang_pairs_map, 
         self.lang_pairs_tree, 
         self.lang_pairs_asp, 
         self.lang_pairs_learned_dist,
         self.lang_embs, 
         self.lang_embs_mapping, # only keys are used to get all supervised languages, no mapping to langembs
         self.iso_lookup) = load_feature_and_embedding_data(learned_dist_path=learned_dist_path)

    def create_csv(self, 
                       feature: str = "learned",
                       zero_shot: bool =False, 
                       n_closest: int = 50,
                       excluded_languages: list = [],
                       excluded_features: list = [],
                       find_furthest: bool = False,
                       individual_distances: bool = False):
        """Create dataset with a given feature's distance in a dict, and saves it to a CSV file."""
        features = ["learned", "map", "tree", "asp", "combined", "random", "oracle"]
        if feature not in features:
            raise ValueError(f"Invalid feature: {feature}. Expected one of: {features}")
        dataset_dict = dict()
        sim_solver = SimilaritySolver(tree_dist=self.lang_pairs_tree, map_dist=self.lang_pairs_map, asp_dict=self.lang_pairs_asp)
        supervised_langs = sorted(self.lang_embs_mapping)
        remove_langs_suffix = ""
        if len(excluded_languages) > 0:
            remove_langs_suffix = "_no-illegal-langs"
            for excl_lang in excluded_languages:
                supervised_langs.remove(excl_lang)
        individual_dist_suffix, excluded_feat_suffix = "", ""
        if feature == "combined":
            if individual_distances:
                individual_dist_suffix = "_indiv-dists"
            if len(excluded_features) > 0:
                excluded_feat_suffix = "_excl-" + "-".join(excluded_features)
        furthest_suffix = "furthest_" if find_furthest else ""                
        zero_shot_suffix= ""
        if zero_shot:
            iso_codes_to_ids = load_json_from_path("iso_lookup.json")[-1]
            zero_shot_suffix = "_zeroshot"
            # remove supervised languages from iso dict
            for sup_lang in supervised_langs:
                iso_codes_to_ids.pop(sup_lang, None)
            lang_codes = list(iso_codes_to_ids)
        else:
            lang_codes = supervised_langs
        failed_langs = []
        if feature == "random":
            random_seed = 0

        for lang in lang_codes:

            if feature == "combined":
                feature_dict = sim_solver.find_closest_multifeature(lang,
                                                            supervised_langs,
                                                            k=n_closest,
                                                            individual_distances=individual_distances,
                                                            excluded_features=excluded_features,
                                                            find_furthest=find_furthest)
            elif feature == "random":
                random_seed += 1
                dataset_dict[lang] = [lang] # target language as first column
                feature_dict = sim_solver.find_closest(feature, 
                                                       lang,
                                                       supervised_langs, 
                                                       k=n_closest,
                                                       find_furthest=find_furthest,
                                                       random_seed=random_seed)                
            else:
                feature_dict = sim_solver.find_closest(feature, 
                                                       lang,
                                                       supervised_langs, 
                                                       k=n_closest,
                                                       find_furthest=find_furthest)

            # discard incomplete results
            if len(feature_dict) < n_closest:
                failed_langs.append(lang)
                continue

            dataset_dict[lang] = [lang] # target language as first column
            # create entry for a single close lang (`feature_dict` must be sorted by distance)
            for _, close_lang in enumerate(feature_dict):
                if feature == "combined":
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
        dataset_columns = ["target_lang"]
        for i in range(n_closest):
            dataset_columns.extend([f"closest_lang_{i}", f"{feature}_dist_{i}"])
            if feature == "combined" and individual_distances:
                if "map" not in excluded_features:
                    dataset_columns.append(f"map_dist_{i}")
                if "asp" not in excluded_features:
                    dataset_columns.append(f"asp_dist_{i}")
                if "tree" not in excluded_features:
                    dataset_columns.append(f"tree_dist_{i}")
        df = pd.DataFrame.from_dict(dataset_dict, orient="index")
        df.columns = dataset_columns
        out_path = os.path.join(DATASET_SAVE_DIR, f"dataset_{feature}_top{n_closest}{zero_shot_suffix}{remove_langs_suffix}{excluded_feat_suffix}{individual_dist_suffix}" + ".csv")
        os.makedirs(DATASET_SAVE_DIR, exist_ok=True)
        df.to_csv(out_path, sep="|", index=False)
        print(f"Failed to retrieve distances for the following languages: {failed_langs}")


def load_feature_and_embedding_data(learned_dist_path=None):
    """Load all features as well as the language embeddings."""
    print("Loading feature and embedding data...")
    with open(LANG_PAIRS_MAP_PATH, "r") as f:
        lang_pairs_map = json.load(f)
    with open(LANG_PAIRS_TREE_PATH, "r") as f:
        lang_pairs_tree = json.load(f)
    if not learned_dist_path:
        learned_dist_path = LANG_PAIRS_LEARNED_DIST_PATH
    with open(learned_dist_path, "r") as f:
        lang_pairs_learned_dist = json.load(f)        
    with open(LANG_PAIRS_ASP_PATH, "rb") as f:
        lang_pairs_asp = pickle.load(f)
    lang_embs = torch.load(LANG_EMBS_PATH)
    with open(LANG_EMBS_MAPPING_PATH, "r") as f:
        lang_embs_mapping = yaml.safe_load(f)
    with open(ISO_LOOKUP_PATH, "r") as f:
        iso_lookup = json.load(f)


    return (lang_pairs_map, lang_pairs_tree, lang_pairs_asp, lang_pairs_learned_dist, lang_embs, lang_embs_mapping, 
            iso_lookup)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learned_dist_path", type=str, default="lang_1_to_lang_2_to_learned_dist_focal4.json", help="filepath of JSON file containing the meta-learned pairwise distances")
    args = parser.parse_args()

    dc = LangDistDatasetCreator(learned_dist_path=args.learned_dist_path)
    excluded_langs = ["deu"]

    dc.create_csv(feature="tree", n_closest=30, zero_shot=False)
    dc.create_csv(feature="map", n_closest=30, zero_shot=False, excluded_languages=excluded_langs)
    dc.create_csv(feature="asp", n_closest=30, zero_shot=False, find_furthest=True)
    dc.create_csv(feature="random", n_closest=30, zero_shot=False, excluded_languages=excluded_langs)
    dc.create_csv(feature="combined", individual_distances=True, n_closest=30, zero_shot=False, excluded_languages=excluded_langs)
    dc.create_csv(feature="learned", n_closest=30, zero_shot=False)
    dc.create_csv(feature="oracle", n_closest=30, zero_shot=False)
