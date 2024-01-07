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

LANGS_IN_TEXT_FRONTEND_PATH = "Preprocessing/multilinguality/languages_from_text_frontend.json"


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


def check_if_language_features_available(lang_pairs_geo, lang_pairs_phylo, lang_pairs_asp, lang_embs_mapping):
    """For each language for which we have a language embedding, check if corresponding features are available"""
    print("Checking if all required features are available...")
    lang_codes = sorted(lang_embs_mapping.keys())
    for lang_code in lang_codes:
        assert lang_pairs_geo[lang_code], f"language code {lang_code} not found in geographic distance file"
        assert lang_pairs_phylo[lang_code], f"language code {lang_code} not found in phylogenetic distance file"
        assert lang_pairs_asp[lang_code] is not None, f"language code {lang_code} not found in ASP file"
    return


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


def check_all_languages_in_text_frontend(languages, lang_pairs_geo, lang_pairs_phylo, lang_pairs_asp, save_path):
    """For all language codes specified in Preprocessing/TextFrontend.py, check if features exist for them.
    Create a dict with all language codes where features are missing, and write it to a JSON file.
    Return the dict that was written to file."""
    geo_errors = []
    phylo_errors = []
    asp_errors = []

    for lang in languages:
        try:
            lang_pairs_geo[lang]
        except KeyError:
            geo_errors.append(lang)
        try:
            lang_pairs_phylo[lang]
        except KeyError:
            phylo_errors.append(lang)
        try:
            lang_pairs_asp[lang]
        except KeyError:
            asp_errors.append(lang)
    
    key_error_dict = {"geo_errors": geo_errors, "phylo_errors": phylo_errors, "asp_errors": asp_errors}
    with open(save_path, "w") as f:
        json.dump(key_error_dict, f)
    return key_error_dict
        

def get_language_pair_features(lang_pairs_geo, lang_pairs_phylo, lang_pairs_asp, lang_embs, lang_embs_mapping, use_phylo=True):
    """Get features for all language-pair combinations."""
    print("Retrieving features for language pairs...")
    features = dict()
    lang_emb_dict = dict()
    languages = sorted(lang_embs_mapping.keys())
    # iterate over all langauges
    for lang_a_idx, lang_a in enumerate(languages):
        print(f"current lang_a: {lang_a}")
        if lang_a_idx < len(languages)-1:
            features[lang_a] = dict()
            # iterate over all remaining languages to get all language pairs
            for lang_b in languages[lang_a_idx+1:]:
                features[lang_a][lang_b] = dict()
                features[lang_a][lang_b]["geo_distance"] = lang_pairs_geo[lang_a][lang_b]
                if use_phylo:
                    try:
                        lang_pair_phylo = lang_pairs_phylo[lang_a][lang_b]
                    except KeyError:
                        lang_pair_phylo = 0
                    features[lang_a][lang_b]["phylo_distance"] = lang_pair_phylo
                features[lang_a][lang_b]["asp"] = asp(lang_a, lang_b, lang_pairs_asp)
        # add language embedding, i.e. the label
        lang_emb_dict[lang_a] = lang_embs[lang_embs_mapping[lang_a]]
    return features, lang_emb_dict


def asp(lang_a, lang_b, path_to_dict):
    """
    Taken from Phat Do.
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


def create_csv(lang_pairs_geo, lang_pairs_phylo, lang_pairs_asp, lang_embs, lang_embs_mapping):
    get_language_pair_features(lang_pairs_geo, lang_pairs_phylo, lang_pairs_asp, lang_embs, lang_embs_mapping)


if __name__ == "__main__":
    lang_pairs_geo, lang_pairs_phylo, lang_pairs_asp, lang_embs, lang_embs_mapping, languages_in_text_frontend = load_feature_and_embedding_data()

    # correct erroneous code for Western Farsi
    lang_embs_mapping = dict((k, v) if k != "fas" else ("pes", v) for k, v in lang_embs_mapping.items())
    languages_in_text_frontend = [l if l != "fas" else "pes" for l in languages_in_text_frontend]

    # for all languages in TextFrontend, check for missing features and write them to file
    key_error_save_path = "Preprocessing/multilinguality/key_errors_for_languages_from_text_frontend.json"
    key_error_dict = check_all_languages_in_text_frontend(languages_in_text_frontend, lang_pairs_geo, lang_pairs_phylo, lang_pairs_asp, key_error_save_path)

    check_if_language_features_available(lang_pairs_geo, lang_pairs_phylo, lang_pairs_asp, lang_embs_mapping)
    feature_dict, lang_emb_dict = get_language_pair_features(lang_pairs_geo, lang_pairs_phylo, lang_pairs_asp, lang_embs, lang_embs_mapping)
    print(feature_dict)
    print(lang_emb_dict.keys())