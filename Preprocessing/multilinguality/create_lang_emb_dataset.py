import numpy as np
import pandas as pd
import os
import json
import yaml
import pickle
import torch
import sys
sys.path.append("/home/behringe/hdd_behringe/IMS-Toucan")
from Preprocessing.TextFrontend import get_language_id

LANG_PAIRS_GEO_PATH = "Preprocessing/multilinguality/lang_1_to_lang_2_to_map_dist.json"
LANG_PAIRS_PHYLO_PATH = "Preprocessing/multilinguality/lang_1_to_lang_2_to_tree_dist.json"
LANG_PAIRS_ASP_PATH = "Preprocessing/multilinguality/asp_dict.pkl"
LANG_EMBS_PATH = "Models/LangEmbs/lang_embs_for_15_languages.pt"
LANG_EMBS_MAPPING_PATH = "Models/LangEmbs/mapping_lang_embs_for_15_languages.yaml"
TEXT_FRONTEND_PATH = "Preprocessing/TextFrontend.py"


# TODO: replace lang_embs_mapping with a function using get_language_id()


class DatasetCreator():
    def __init__(self):
        (self.lang_pairs_geo, 
         self.lang_pairs_phylo, 
         self.lang_pairs_asp, 
         self.lang_embs, 
         self.lang_embs_mapping, 
         self.languages_in_text_frontend) = load_feature_and_embedding_data()
        
        # correct erroneous code for Western Farsi
        self.lang_embs_mapping = dict((k, v) if k != "fas" else ("pes", v) for k, v in self.lang_embs_mapping.items())
        self.languages_in_text_frontend = [l if l != "fas" else "pes" for l in self.languages_in_text_frontend]

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

    def get_features_for_one_language(self, specified_language, use_phylo=True):
        """Get all features for one specific language code"""
        feature_dict = {"geo_distance": [], "phylo_distance": [], "asp": []}
        languages = sorted(self.lang_embs_mapping.keys())
        specified_lang_idx = languages.index(specified_language) # index of the desired language

        # get all pairwise features
        for idx, other_lang in enumerate(languages):
            if idx <= specified_lang_idx:
                feature_dict["geo_distance"].append(self.lang_pairs_geo[other_lang][specified_language])
                if use_phylo:
                    try:
                        lang_pair_phylo = self.lang_pairs_phylo[other_lang][specified_language]
                    except KeyError:
                        lang_pair_phylo = 0
                    feature_dict["phylo_distance"].append(lang_pair_phylo)
                feature_dict["asp"].append(asp(other_lang, specified_language, self.lang_pairs_asp))
            else:
                feature_dict["geo_distance"].append(self.lang_pairs_geo[specified_language][other_lang])
                if use_phylo:
                    try:
                        lang_pair_phylo = self.lang_pairs_phylo[specified_language][other_lang]
                    except KeyError:
                        lang_pair_phylo = 0
                    feature_dict["phylo_distance"].append(lang_pair_phylo)
                feature_dict["asp"].append(asp(specified_language, other_lang, self.lang_pairs_asp))

        return feature_dict

    def create_json(self, use_phylo=True):
        dataset_dict = dict()
        for lang in sorted(self.lang_embs_mapping.keys()):
            feature_dict = self.get_features_for_one_language(lang, use_phylo=use_phylo)
            lang_emb = self.lang_embs[self.lang_embs_mapping[lang]]
            dataset_dict[lang] = []
            for feat in feature_dict.keys():
                dataset_dict[lang].append(np.asarray(feature_dict[feat]))
            dataset_dict[lang].append(lang_emb.numpy())

        dataset_columns = []
        dataset_columns.extend(feature_dict.keys())
        dataset_columns.append("language_embedding")
        df = pd.DataFrame.from_dict(dataset_dict, orient="index")
        df.index.name = "language"
        df.columns = dataset_columns
        df.to_json("/home/behringe/hdd_behringe/IMS-Toucan/Preprocessing/multilinguality/dataset.json")


def asp(lang_a, lang_b, path_to_dict):
    """
    Based on Phat Do's code.
    Look up and return the ASP between lang_a and lang_b from (pre-calculated) dictionary at path_to_dict
    """
    if isinstance(path_to_dict, dict):
        asp_dict = path_to_dict
    else:
        with open(path_to_dict, 'rb') as dictfile:
            asp_dict = pickle.load(dictfile)	

    lang_list = list(asp_dict.keys()) # list of all languages, to get lang_b's index
    lang_b_idx = lang_list.index(lang_b) # lang_b's index
    asp = asp_dict[lang_a][lang_b_idx] # asp_dict's structure: {lang: numpy array of all corresponding ASPs}
    
    return asp

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

    languages_in_text_frontend = get_languages_from_text_frontend()

    return lang_pairs_geo, lang_pairs_phylo, lang_pairs_asp, lang_embs, lang_embs_mapping, languages_in_text_frontend


if __name__ == "__main__":

    dc = DatasetCreator()

    key_error_save_path = "Preprocessing/multilinguality/key_errors_for_languages_from_text_frontend.json"
    key_error_dict = dc.check_all_languages_in_text_frontend(save_path=key_error_save_path)

    # feature_dict, lang_emb_dict = dc.get_language_pair_features()

    dc.create_json()
