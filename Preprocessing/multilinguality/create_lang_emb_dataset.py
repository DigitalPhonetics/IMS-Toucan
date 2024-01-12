import numpy as np
import pandas as pd
import os
import json
import yaml
import pickle
import torch
from tqdm import tqdm
from copy import deepcopy
import sys
sys.path.append("/home/behringe/hdd_behringe/IMS-Toucan")
from Preprocessing.TextFrontend import get_language_id
from Preprocessing.multilinguality.SimilaritySolver import SimilaritySolver
from Preprocessing.multilinguality.asp import asp, load_asp_dict

ISO_LOOKUP_PATH = "iso_lookup.json"
ISO_TO_FULLNAME_PATH = "iso_to_fullname.json"
LANG_PAIRS_GEO_PATH = "lang_1_to_lang_2_to_map_dist.json"
LANG_PAIRS_PHYLO_PATH = "lang_1_to_lang_2_to_tree_dist.json"
LANG_PAIRS_ASP_PATH = "asp_dict.pkl"
LANG_EMBS_PATH = "LangEmbs/lang_embs_for_15_languages.pt"
LANG_EMBS_MAPPING_PATH = "LangEmbs/mapping_lang_embs_for_15_languages.yaml"
TEXT_FRONTEND_PATH = "../TextFrontend.py"


# TODO: replace lang_embs_mapping with a function using get_language_id()


class DatasetCreator():
    def __init__(self):
        (self.lang_pairs_geo, 
         self.lang_pairs_phylo, 
         self.lang_pairs_asp, 
         self.lang_embs, 
         self.lang_embs_mapping, 
         self.languages_in_text_frontend,
         self.iso_lookup) = load_feature_and_embedding_data()
        
    def check_if_language_features_available(self):
        """For each language for which we have a language embedding, check if corresponding features are available"""
        print("Checking if all required features are available...")
        lang_codes = sorted(self.lang_embs_mapping.keys())
        for lang_code in lang_codes:
            assert self.lang_pairs_geo[lang_code], f"language code {lang_code} not found in geographic distance file"
            assert self.lang_pairs_phylo[lang_code], f"language code {lang_code} not found in phylogenetic distance file"
            assert self.lang_pairs_asp[lang_code] is not None, f"language code {lang_code} not found in ASP file"
        return

    def check_all_languages_in_text_frontend(self, save_path):
        """For all language codes specified in Preprocessing/TextFrontend.py, check if features exist for them.
        Create a dict with all language codes where features are missing, and write it to a JSON file.
        Return the dict that was written to file."""
        geo_errors = []
        phylo_errors = []
        asp_errors = []

        for lang in self.languages_in_text_frontend:
            try:
                self.lang_pairs_geo[lang]
            except KeyError:
                geo_errors.append(lang)
            try:
                self.lang_pairs_phylo[lang]
            except KeyError:
                phylo_errors.append(lang)
            try:
                self.lang_pairs_asp[lang]
            except KeyError:
                asp_errors.append(lang)
        
        key_error_dict = {"geo_errors": geo_errors, "phylo_errors": phylo_errors, "asp_errors": asp_errors}
        with open(save_path, "w") as f:
            json.dump(key_error_dict, f)
        return key_error_dict

    def get_language_pair_features(self, use_phylo=True):
        """Get features for all language-pair combinations."""
        print("Retrieving features for language pairs...")
        feature_dict = dict()
        lang_emb_dict = dict()
        languages = sorted(self.lang_embs_mapping.keys())
        # iterate over all langauges
        for lang_a_idx, lang_a in enumerate(languages):
            if lang_a_idx < len(languages)-1:
                feature_dict[lang_a] = dict()
                # iterate over all remaining languages to get all language pairs
                for lang_b in languages[lang_a_idx+1:]:
                    feature_dict[lang_a][lang_b] = dict()
                    feature_dict[lang_a][lang_b]["geo_distance"] = self.lang_pairs_geo[lang_a][lang_b]
                    if use_phylo:
                        try:
                            lang_pair_phylo = self.lang_pairs_phylo[lang_a][lang_b]
                        except KeyError:
                            lang_pair_phylo = 0
                        feature_dict[lang_a][lang_b]["phylo_distance"] = lang_pair_phylo
                    feature_dict[lang_a][lang_b]["asp"] = asp(lang_a, lang_b, self.lang_pairs_asp)
            # add language embedding, i.e. the label
            lang_emb_dict[lang_a] = self.lang_embs[self.lang_embs_mapping[lang_a]]
        return feature_dict, lang_emb_dict

    # def get_features_for_one_language(self, specified_language, languages, use_phylo=True):
    #     """Get all features for one specific language code"""
    #     feature_dict = {"geo_distance": [], "phylo_distance": [], "asp": []}
    #     specified_lang_idx = languages.index(specified_language) # index of the desired language

    #     # get all pairwise features
    #     for idx, other_lang in enumerate(languages):
    #         if idx <= specified_lang_idx:
    #             lang_a, lang_b = other_lang, specified_language
    #         else:
    #             lang_a, lang_b = specified_language, other_lang
    #         feature_dict["geo_distance"].append(self.lang_pairs_geo[lang_a][lang_b])
    #         if use_phylo:
    #             try:
    #                 lang_pair_phylo = self.lang_pairs_phylo[lang_a][lang_b]
    #             except KeyError:
    #                 lang_pair_phylo = 0
    #             feature_dict["phylo_distance"].append(lang_pair_phylo)
    #         asp_return = asp(lang_a, lang_b, self.lang_pairs_asp)
    #         if isinstance(asp_return, ValueError) or isinstance(asp_return, KeyError):
    #             return asp_return
    #         feature_dict["asp"].append(asp_return)
    #     return feature_dict
    def get_features_for_one_language(self, sim_solver: SimilaritySolver, specified_language, languages, n_closest, use_phylo=True):
        """Get features for one specific language code"""
        # feature_dict = {"geo_distance": [], "phylo_distance": [], "asp": []}
        feature_dict = dict()

        # get all pairwise features

        # find n closest languages which should be used for features in the dataset
        closest_langs_on_map = sim_solver.find_closest_on_map(lang=specified_language, supervised_langs=languages, n_closest=n_closest)
        for idx, other_lang in enumerate(closest_langs_on_map):
            # assign feature to dict
            try:
                feature_dict[f"geo_distance_{idx}"] = [self.lang_pairs_geo[specified_language][other_lang]]
            except KeyError:
                feature_dict[f"geo_distance_{idx}"] = [self.lang_pairs_geo[other_lang][specified_language]]
            # append language embedding to feature
            feature_dict[f"geo_distance_{idx}"].extend(self.lang_embs[self.lang_embs_mapping[other_lang]].numpy())

        if use_phylo:
            closest_langs_in_family = sim_solver.find_closest_in_family(lang=specified_language, supervised_langs=languages, n_closest=n_closest)

            for idx, other_lang in enumerate(closest_langs_in_family):
                try:
                    lang_pair_phylo = self.lang_pairs_phylo[specified_language][other_lang]
                except KeyError:
                    try:
                        lang_pair_phylo = self.lang_pairs_phylo[other_lang][specified_language]
                    except KeyError:
                        lang_pair_phylo = 0
                feature_dict[f"phylo_distance_{idx}"] = [lang_pair_phylo]
                feature_dict[f"phylo_distance_{idx}"].extend(self.lang_embs[self.lang_embs_mapping[other_lang]].numpy())

        closest_langs_aspf = sim_solver.find_closest_aspf(specified_language, languages, n_closest=n_closest)
        for idx, other_lang in enumerate(closest_langs_aspf):
            feature_dict[f"asp_{idx}"] = [asp(specified_language, other_lang, self.lang_pairs_asp)]
            feature_dict[f"asp_{idx}"].extend(self.lang_embs[self.lang_embs_mapping[other_lang]].numpy())

        return feature_dict


    def create_json(self, n_closest=5, use_phylo=True):
        """Create dataset in a dict, and saves it to a JSON file."""
        dataset_dict = dict()
        # TODO: create smaller lookup dicts containing only the values for the currently used languages 
        sim_solver = SimilaritySolver(tree_dist=self.lang_pairs_phylo, map_dist=self.lang_pairs_geo, asp_dict=self.lang_pairs_asp)
        for lang in sorted(self.lang_embs_mapping.keys()):
            feature_dict = self.get_features_for_one_language(sim_solver, lang, sorted(self.lang_embs_mapping.keys()), n_closest=n_closest, use_phylo=use_phylo)
            y_lang_emb = self.lang_embs[self.lang_embs_mapping[lang]]
            dataset_dict[lang] = []
            for feat in feature_dict.keys():
                dataset_dict[lang].append(np.asarray(feature_dict[feat]))
            dataset_dict[lang].append(y_lang_emb.numpy())

        dataset_columns = list(feature_dict.keys())
        dataset_columns.append("language_embedding")
        df = pd.DataFrame.from_dict(dataset_dict, orient="index")
        df.index.name = "language"
        df.columns = dataset_columns
        df.to_json("/home/behringe/hdd_behringe/IMS-Toucan/Preprocessing/multilinguality/dataset.json")


    def create_1D_json(self, n_closest=5, use_phylo=True):
        """Create dataset in a dict, and saves it to a JSON file."""
        dataset_dict = dict()
        # TODO: create smaller lookup dicts containing only the values for the currently used languages 
        sim_solver = SimilaritySolver(tree_dist=self.lang_pairs_phylo, map_dist=self.lang_pairs_geo, asp_dict=self.lang_pairs_asp)
        for lang in sorted(self.lang_embs_mapping.keys()):
            feature_dict = self.get_features_for_one_language(sim_solver, lang, sorted(self.lang_embs_mapping.keys()), n_closest=n_closest, use_phylo=use_phylo)
            y_lang_emb = self.lang_embs[self.lang_embs_mapping[lang]].numpy()
            for dim in range(y_lang_emb.size):
                lang_dim_key = f"{lang}_{dim}"
                dataset_dict[lang_dim_key] = [dim]
                for feat in feature_dict.keys():

                    dataset_dict[lang_dim_key].append(feature_dict[feat][0]) # get feature, e.g. geo_distance of closest lang
                    dataset_dict[lang_dim_key].append(feature_dict[feat][dim+1]) # get 1 dimension of corresponding lang's lang emb
                dataset_dict[lang_dim_key].append(y_lang_emb[dim]) # get target, i.e. 1 dimension of target lang emb

        dataset_columns = ["dim"]
        for key in feature_dict.keys():
            dataset_columns.append(f"{key}_score")
            dataset_columns.append(f"{key}_emb_dim")
        dataset_columns.append("language_embedding")
        df = pd.DataFrame.from_dict(dataset_dict, orient="index")
        print(df)
        df.index.name = "language"
        df.columns = dataset_columns
        df.to_json("/home/behringe/hdd_behringe/IMS-Toucan/Preprocessing/multilinguality/dataset_1D.json")

    def check_features_for_all_languages(self):
        """Check feature generation for all language combinations.
        Write all erroneous ISO codes to a file."""

        print("Checking features for all language combinations. This may take a while.")
        # get all iso codes
        iso_codes = list(self.iso_lookup[0].keys())
        # missing keys were retrieved via get_language_diff_asp_iso_lookup.py
        # TODO: integrate that file into this file
        missing_keys = ["ajp", "ajt", "en-sc", "en-us", "fr-be", "fr-sw", "lak", "lno", "nul", "pii", "plj", "pt-br", "slq", "smd", "snb", "spa-lat", "tpw", "vi-ctr", "vi-so", "wya", "zua"]
        for k in missing_keys:
            try:
                iso_codes.remove(k)
            except:
                print(f"{k} not in iso_codes")
        erroneous_codes = []
        for code in tqdm(iso_codes, desc="Generating features"):
            try:
                dc.get_features_for_one_language(code, iso_codes)
            except:
                print(f"Error when processing code {code}")
                erroneous_codes.append(code)
        print(f"Generating features failed for {len(erroneous_codes)}/{len(iso_codes)} language codes.")
        erroneous_codes_save_path = "erroneous_iso_codes.json"
        with open(erroneous_codes_save_path, "w") as f:
            json.dump(erroneous_codes, f)


# def asp(lang_a, lang_b, path_to_dict):
#     """
#     Based on Phat Do's code.
#     Look up and return the ASP between lang_a and lang_b from (pre-calculated) dictionary at path_to_dict
#     """
#     asp_dict = load_asp_dict(path_to_dict)
#
#     lang_list = list(asp_dict.keys()) # list of all languages, to get lang_b's index
#     try:
#         lang_b_idx = lang_list.index(lang_b) # lang_b's index
#     except:
#         return ValueError(lang_b)
#     try:
#         asp = asp_dict[lang_a][lang_b_idx] # asp_dict's structure: {lang: numpy array of all corresponding ASPs}
#     except:
#         return KeyError(lang_a)
    
#     return asp


def get_languages_from_text_frontend(filepath=TEXT_FRONTEND_PATH):
    """Load TextFrontend.py and extract all ISO 639-2 language codes for which G2P rules exist.
    Return a list containing all extracted languages."""
    with open(filepath, "r") as f:
        lines = f.readlines()
    languages = []
    for line in lines:
        if "if language == " in line:
            languages.append(line.split('"')[1])
    return languages

def load_feature_and_embedding_data():
    """Load all features as well as the language embeddings."""
    print("Loading feature and embedding data...")
    with open(LANG_PAIRS_GEO_PATH, "r") as f:
        lang_pairs_geo = json.load(f)
    with open(LANG_PAIRS_PHYLO_PATH, "r") as f:
        lang_pairs_phylo = json.load(f)
    with open(LANG_PAIRS_ASP_PATH, "rb") as f:
        lang_pairs_asp = pickle.load(f)
    lang_embs = torch.load(LANG_EMBS_PATH)
    with open(LANG_EMBS_MAPPING_PATH, "r") as f:
        lang_embs_mapping = yaml.safe_load(f)
    with open(ISO_LOOKUP_PATH, "r") as f:
        iso_lookup = json.load(f)
    languages_in_text_frontend = get_languages_from_text_frontend()


    return lang_pairs_geo, lang_pairs_phylo, lang_pairs_asp, lang_embs, lang_embs_mapping, languages_in_text_frontend, iso_lookup




if __name__ == "__main__":

    dc = DatasetCreator()

    # key_error_save_path = "Preprocessing/multilinguality/key_errors_for_languages_from_text_frontend.json"
    # key_error_dict = dc.check_all_languages_in_text_frontend(save_path=key_error_save_path)

    # # feature_dict, lang_emb_dict = dc.get_language_pair_features()

    # dc.create_json()
    dc.create_1D_json()

    check_features_for_all_languages = False
    if check_features_for_all_languages:
        dc.check_features_for_all_languages()
