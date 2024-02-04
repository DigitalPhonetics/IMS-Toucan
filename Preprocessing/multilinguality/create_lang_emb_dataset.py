import numpy as np
import pandas as pd
import os
import random
import json
import yaml
import pickle
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from copy import deepcopy
import sys
sys.path.append("/home/behringe/hdd_behringe/IMS-Toucan")
from Preprocessing.TextFrontend import get_language_id, load_json_from_path
from Preprocessing.multilinguality.SimilaritySolver import SimilaritySolver
from Preprocessing.multilinguality.asp import asp, load_asp_dict

ISO_LOOKUP_PATH = "iso_lookup.json"
ISO_TO_FULLNAME_PATH = "iso_to_fullname.json"
LANG_PAIRS_map_PATH = "lang_1_to_lang_2_to_map_dist.json"
LANG_PAIRS_tree_PATH = "lang_1_to_lang_2_to_tree_dist.json"
LANG_PAIRS_ASP_PATH = "asp_dict.pkl"
NUM_LANGS = 463
LOSS_TYPE = "with_less_loss_fixed_tree_distance"
LANG_EMBS_PATH = f"LangEmbs/final_model_{LOSS_TYPE}.pt"

LANG_EMBS_MAPPING_PATH = f"LangEmbs/mapping_lang_embs_{NUM_LANGS}_langs.yaml"
# TODO: get lang_embs in a nicer way than from this mapping
JSON_OUT_PATH = f"/home/behringe/hdd_behringe/IMS-Toucan/Preprocessing/multilinguality/datasets/dataset_{NUM_LANGS}_{LOSS_TYPE}.json"
JSON_1D_OUT_PATH = f"/home/behringe/hdd_behringe/IMS-Toucan/Preprocessing/multilinguality/datasets/dataset_1D_{NUM_LANGS}_{LOSS_TYPE}.json"
ASP_CSV_OUT_PATH = f"/home/behringe/hdd_behringe/IMS-Toucan/Preprocessing/multilinguality/datasets/dataset_asp_{NUM_LANGS}_{LOSS_TYPE}.csv"
MAP_CSV_OUT_PATH = f"/home/behringe/hdd_behringe/IMS-Toucan/Preprocessing/multilinguality/datasets/dataset_map_{NUM_LANGS}_{LOSS_TYPE}.csv"
TREE_CSV_OUT_PATH = f"/home/behringe/hdd_behringe/IMS-Toucan/Preprocessing/multilinguality/datasets/dataset_tree_{NUM_LANGS}_{LOSS_TYPE}.csv"
COMBINED_CSV_OUT_PATH = f"/home/behringe/hdd_behringe/IMS-Toucan/Preprocessing/multilinguality/datasets/dataset_COMBINED_{NUM_LANGS}_{LOSS_TYPE}.csv"
RANDOM_CSV_OUT_PATH = f"/home/behringe/hdd_behringe/IMS-Toucan/Preprocessing/multilinguality/datasets/dataset_random_{NUM_LANGS}_{LOSS_TYPE}.csv"
TEXT_FRONTEND_PATH = "../TextFrontend.py"


class DatasetCreator():
    def __init__(self):
        (self.lang_pairs_map, 
         self.lang_pairs_tree, 
         self.lang_pairs_asp, 
         self.lang_embs, 
         self.lang_embs_mapping, # only keys are used to get all supervised languages, no mapping to langembs
         self.languages_in_text_frontend,
         self.iso_lookup) = load_feature_and_embedding_data()


    def get_features_for_one_language(self, sim_solver: SimilaritySolver, specified_language, languages, n_closest, use_tree=True):
        """Get features for one specific language code"""
        # feature_dict = {"map_distance": [], "tree_distance": [], "asp": []}
        feature_dict = dict()

        # get all pairwise features

        # find n closest languages which should be used for features in the dataset
        closest_langs_on_map = sim_solver.find_closest_on_map(lang=specified_language, supervised_langs=languages, n_closest=n_closest)
        for idx, other_lang in enumerate(closest_langs_on_map):
            # assign feature to dict
            try:
                feature_dict[f"map_distance_{idx}"] = [self.lang_pairs_map[specified_language][other_lang]]
            except KeyError:
                feature_dict[f"map_distance_{idx}"] = [self.lang_pairs_map[other_lang][specified_language]]
            # append language embedding to feature
            feature_dict[f"map_distance_{idx}"].extend(self.lang_embs[self.lang_embs_mapping[other_lang]].numpy())

        if use_tree:
            closest_langs_in_family = sim_solver.find_closest_in_family(lang=specified_language, supervised_langs=languages, n_closest=n_closest)

            for idx, other_lang in enumerate(closest_langs_in_family):
                try:
                    lang_pair_tree = self.lang_pairs_tree[specified_language][other_lang]
                except KeyError:
                    try:
                        lang_pair_tree = self.lang_pairs_tree[other_lang][specified_language]
                    except KeyError:
                        lang_pair_tree = 0
                feature_dict[f"tree_distance_{idx}"] = [lang_pair_tree]
                feature_dict[f"tree_distance_{idx}"].extend(self.lang_embs[self.lang_embs_mapping[other_lang]].numpy())

        closest_langs_aspf = sim_solver.find_closest_aspf(specified_language, languages, n_closest=n_closest)
        for idx, other_lang in enumerate(closest_langs_aspf):
            feature_dict[f"asp_{idx}"] = [asp(specified_language, other_lang, self.lang_pairs_asp)]
            feature_dict[f"asp_{idx}"].extend(self.lang_embs[self.lang_embs_mapping[other_lang]].numpy())

        return feature_dict


    def create_combined_csv(self, distance_type="average", zero_shot=False, individual_distances=False, n_closest=5):
        """Create dataset (with combined Euclidean distance) in a dict, and saves it to a JSON file."""
        dataset_dict = dict()
        sim_solver = SimilaritySolver(tree_dist=self.lang_pairs_tree, map_dist=self.lang_pairs_map, asp_dict=self.lang_pairs_asp)
        distance_type = distance_type
        supervised_langs = sorted(list(self.lang_embs_mapping.keys()))
        illegal_langs = ["deu", "eng"]
        for il_lang in illegal_langs:
            supervised_langs.remove(il_lang)
        individual_dist_suffix = "_individual_dists" if individual_distances else ""
        zero_shot_suffix= ""
        if zero_shot:
            iso_codes_to_ids = load_json_from_path("iso_lookup.json")[-1]
            zero_shot_suffix = "zero_shot_"
            # remove supervised languages from iso dict
            for sup_lang in supervised_langs:
                iso_codes_to_ids.pop(sup_lang, None)
            lang_codes = list(iso_codes_to_ids.keys())
        else:
            lang_codes = sorted(self.lang_embs_mapping.keys())
        failed_langs = []
        for lang in lang_codes:
            dataset_dict[lang] = [lang] # target language as first column
            feature_dict = sim_solver.find_closest_combined(lang, 
                                                            supervised_langs, 
                                                            distance=distance_type, 
                                                            n_closest=n_closest,
                                                            individual_distances=individual_distances)
            # sort out incomplete results
            if len(feature_dict) < n_closest:
                failed_langs.append(lang)
                continue
            # create entry for a single close lang
            for _, close_lang in enumerate(feature_dict):
                close_lang_euclid = feature_dict[close_lang]["combined_distance"]
                close_lang_feature_list = [close_lang, close_lang_euclid]
                if individual_distances:
                    close_lang_distances = feature_dict[close_lang]["individual_distances"]
                    close_lang_feature_list.extend(close_lang_distances)
                # column order: compared closest language, euclid_dist, map_dist, tree_dist, asp_dist
                dataset_dict[lang].extend(close_lang_feature_list)
        dataset_columns = ["target_lang"]
        for i in range(n_closest):
            dataset_columns.extend([f"closest_lang_{i}", f"{distance_type}_dist_{i}"])
            if individual_distances:
                dataset_columns.extend([f"map_dist_{i}", f"tree_dist_{i}", f"asp_dist_{i}"])
        df = pd.DataFrame.from_dict(dataset_dict, orient="index")
        df.columns = dataset_columns
        out_path = COMBINED_CSV_OUT_PATH.split(".")[0] + f"_{zero_shot_suffix}{distance_type}{individual_dist_suffix}" + ".csv"
        df.to_csv(out_path, sep="|", index=False)
        print(f"Failed to retrieve scores for the following languages: {failed_langs}")

    def create_random_csv(self, n=5):
        """Create dataset with randomly assigned `closest` languages, and each similarity/distance value set to 0.5."""
        dataset_dict = dict()
        sim_solver = SimilaritySolver(tree_dist=self.lang_pairs_tree, map_dist=self.lang_pairs_map, asp_dict=self.lang_pairs_asp)
        supervised_langs = sorted(self.lang_embs_mapping.keys())
        random.seed(42)
        cumulative_seed = 0
        for lang in supervised_langs:
            cumulative_seed += 1
            dataset_dict[lang] = [lang]
            feature_dict = sim_solver.get_random_languages(lang,
                                                           supervised_langs,
                                                           n=n,
                                                           random_seed=cumulative_seed)
            for _, close_lang in enumerate(feature_dict):
                score = feature_dict[close_lang]
                # column order: compared closest language, asp_dist
                close_lang_feature_list = [close_lang, score]
                dataset_dict[lang].extend(close_lang_feature_list)
        dataset_columns = ["target_lang"]
        for i in range(n):
            dataset_columns.extend([f"closest_lang_{i}", f"fixed_dist_{i}"])
        df = pd.DataFrame.from_dict(dataset_dict, orient="index")
        df.columns = dataset_columns
        df.to_csv(RANDOM_CSV_OUT_PATH, sep="|", index=False)

    def create_aspf_csv(self, zero_shot=False, n_closest=5):
        """Create dataset (with combined Euclidean distance) in a dict, and saves it to a JSON file."""
        dataset_dict = dict()
        sim_solver = SimilaritySolver(tree_dist=self.lang_pairs_tree, map_dist=self.lang_pairs_map, asp_dict=self.lang_pairs_asp)
        supervised_langs = sorted(self.lang_embs_mapping.keys())
        zero_shot_suffix= ""
        if zero_shot:
            iso_codes_to_ids = load_json_from_path("iso_lookup.json")[-1]
            zero_shot_suffix = "_zero_shot"
            # remove supervised languages from iso dict
            for sup_lang in supervised_langs:
                iso_codes_to_ids.pop(sup_lang, None)
            lang_codes = list(iso_codes_to_ids.keys())
        else:
            lang_codes = sorted(self.lang_embs_mapping.keys())
        failed_langs = []        
        for lang in lang_codes:
            dataset_dict[lang] = [lang] # target language as first column
            feature_dict = sim_solver.find_closest_aspf(lang, 
                                                        supervised_langs, 
                                                        n_closest=n_closest)
            # sort out incomplete results
            if len(feature_dict) < n_closest:
                failed_langs.append(lang)
                continue            
            # create entry for a single close lang
            for _, close_lang in enumerate(feature_dict):
                score = feature_dict[close_lang]
                # column order: compared closest language, asp_dist
                close_lang_feature_list = [close_lang, score]
                dataset_dict[lang].extend(close_lang_feature_list)
        dataset_columns = ["target_lang"]
        for i in range(n_closest):
            dataset_columns.extend([f"closest_lang_{i}", f"asp_dist_{i}"])
        df = pd.DataFrame.from_dict(dataset_dict, orient="index")
        df.columns = dataset_columns
        out_path = ASP_CSV_OUT_PATH.split(".")[0] + f"{zero_shot_suffix}" + ".csv"
        df.to_csv(out_path, sep="|", index=False)
        print(f"Failed to retrieve scores for the following languages: {failed_langs}")

    def create_map_csv(self, zero_shot=False, n_closest=5):
        """Create dataset (with combined Euclidean distance) in a dict, and saves it to a JSON file."""
        dataset_dict = dict()
        sim_solver = SimilaritySolver(tree_dist=self.lang_pairs_tree, map_dist=self.lang_pairs_map, asp_dict=self.lang_pairs_asp)
        supervised_langs = sorted(self.lang_embs_mapping.keys())
        zero_shot_suffix= ""
        if zero_shot:
            iso_codes_to_ids = load_json_from_path("iso_lookup.json")[-1]
            zero_shot_suffix = "_zero_shot"
            # remove supervised languages from iso dict
            for sup_lang in supervised_langs:
                iso_codes_to_ids.pop(sup_lang, None)
            lang_codes = list(iso_codes_to_ids.keys())
        else:
            lang_codes = sorted(self.lang_embs_mapping.keys())
        failed_langs = []        
        for lang in lang_codes:
            dataset_dict[lang] = [lang] # target language as first column
            feature_dict = sim_solver.find_closest_on_map(lang, 
                                                        supervised_langs,
                                                        n_closest=n_closest)
            # sort out incomplete results
            if len(feature_dict) < n_closest:
                failed_langs.append(lang)
                continue            
            # create entry for a single close lang
            for _, close_lang in enumerate(feature_dict):
                score = feature_dict[close_lang]
                # column order: compared closest language, asp_dist
                close_lang_feature_list = [close_lang, score]
                dataset_dict[lang].extend(close_lang_feature_list)
        dataset_columns = ["target_lang"]
        for i in range(n_closest):
            dataset_columns.extend([f"closest_lang_{i}", f"map_dist_{i}"])
        df = pd.DataFrame.from_dict(dataset_dict, orient="index")
        df.columns = dataset_columns
        out_path = MAP_CSV_OUT_PATH.split(".")[0] + f"{zero_shot_suffix}" + ".csv"
        df.to_csv(out_path, sep="|", index=False)
        print(f"Failed to retrieve scores for the following languages: {failed_langs}")


    def create_tree_csv(self, zero_shot=False, n_closest=5):
        """Create dataset (with combined Euclidean distance) in a dict, and saves it to a JSON file."""
        dataset_dict = dict()
        sim_solver = SimilaritySolver(tree_dist=self.lang_pairs_tree, map_dist=self.lang_pairs_map, asp_dict=self.lang_pairs_asp)
        supervised_langs = sorted(self.lang_embs_mapping.keys())
        zero_shot_suffix= ""
        if zero_shot:
            iso_codes_to_ids = load_json_from_path("iso_lookup.json")[-1]
            zero_shot_suffix = "_zero_shot"
            # remove supervised languages from iso dict
            for sup_lang in supervised_langs:
                iso_codes_to_ids.pop(sup_lang, None)
            lang_codes = list(iso_codes_to_ids.keys())
        else:
            lang_codes = sorted(self.lang_embs_mapping.keys())
        failed_langs = []
        for lang in lang_codes:
            dataset_dict[lang] = [lang] # target language as first column
            feature_dict = sim_solver.find_closest_in_family(lang,
                                                        supervised_langs,
                                                        n_closest=n_closest)
            # sort out incomplete results
            if len(feature_dict) < n_closest:
                failed_langs.append(lang)
                continue            
            # create entry for a single close lang
            for _, close_lang in enumerate(feature_dict):
                score = feature_dict[close_lang]
                # column order: compared closest language, asp_dist
                close_lang_feature_list = [close_lang, score]
                dataset_dict[lang].extend(close_lang_feature_list)
        dataset_columns = ["target_lang"]
        for i in range(n_closest):
            dataset_columns.extend([f"closest_lang_{i}", f"tree_dist_{i}"])
        df = pd.DataFrame.from_dict(dataset_dict, orient="index")
        df.columns = dataset_columns
        out_path = TREE_CSV_OUT_PATH.split(".")[0] + f"{zero_shot_suffix}" + ".csv"
        df.to_csv(out_path, sep="|", index=False)
        print(f"Failed to retrieve scores for the following languages: {failed_langs}")

    def create_feature_csv_from_lookup_csv(self, csv_path, out_path, single_dim=False):
        """Takes as input a dataset CSV containing only the ISO codes of the closest languages 
        (i.e. the actual features still need to be looked up), and creates a new dataset CSV that 
        contains the actual language embedding features of the closest languages.
        If single_dim is set to True, creates one csv file for each dimension of the language embeddings.
        """

        df = pd.read_csv(csv_path, sep="|")
        if single_dim:
            lang_emb_dims = self.lang_embs.shape[1]
            dim_specific_dicts = {dim: {} for dim in range(lang_emb_dims)}
        else:
            new_dataset_dict = {}
        for row in df.itertuples():
            features_for_one_sample = []
            target_lang = getattr(row, "target_lang")
            for i, val in enumerate(row):
                # use index as key for dict
                if i == 0 and val == row.Index:
                    row_key = val
                    continue
                # strings are lang codes and need to be looked up
                if isinstance(val, str):
                    if val == target_lang: # ignore target_lang for now and add it at the end
                        continue
                    lang_emb = self.lang_embs[get_language_id(val)].squeeze(0).numpy()
                    lang_emb = lang_emb.tolist()
                    if single_dim:
                        for dim_key, dim_val in enumerate(lang_emb):
                            if not row_key in dim_specific_dicts[dim_key]:
                                dim_specific_dicts[dim_key][row_key] = [dim_val]
                            else:
                                dim_specific_dicts[dim_key][row_key].append(dim_val)
                    else:
                        features_for_one_sample.extend(lang_emb)

                else:
                    # distance value as float
                    if single_dim: 
                        for dim_key in dim_specific_dicts:
                            dim_specific_dicts[dim_key][row_key].append(val)
                    else:
                        features_for_one_sample.append(val)
            # add target lang at the end of each row
            target_lang_emb = self.lang_embs[get_language_id(row.target_lang)].squeeze(0).numpy()
            target_lang_emb = target_lang_emb.tolist()
            if single_dim:
                for dim_key, dim_val in enumerate(target_lang_emb):
                    dim_specific_dicts[dim_key][row_key].append(dim_val)
            else:
                features_for_one_sample.extend(target_lang_emb)
                # add all features of one row to the dict
                new_dataset_dict[row_key] = features_for_one_sample
        if single_dim:
            for dim_key, dim_dict in enumerate(dim_specific_dicts):
                df = pd.DataFrame.from_dict(dim_specific_dicts[dim_dict], orient="index")
                out_path_prefix, out_path_ext = out_path.split(".")
                df.to_csv(f"{out_path_prefix}_dim{dim_key}.{out_path_ext}", sep="|", index=False)
        else:
            # create the dataframe
            dataset_with_features_df = pd.DataFrame.from_dict(new_dataset_dict, orient="index")
            # write to file
            dataset_with_features_df.to_csv(out_path, sep="|", index=False)


def create_repeated_df(df, out_path=None, n_repeats=100):
    """
    Create repeated df which can then be used for noise augmentation.
    Return the repeated df.
    """
    # shuffle before repeating
    new_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # repeat
    new_df = pd.DataFrame(np.repeat(new_df.values, repeats=n_repeats, axis=0))
    new_df.columns = df.columns
    if out_path:
        new_df.to_csv(out_path, sep="|")
    return new_df


    def create_json(self, n_closest=5, use_tree=True):
        """Create dataset in a dict, and saves it to a JSON file."""
        dataset_dict = dict()
        # TODO: create smaller lookup dicts containing only the values for the currently used languages 
        sim_solver = SimilaritySolver(tree_dist=self.lang_pairs_tree, map_dist=self.lang_pairs_map, asp_dict=self.lang_pairs_asp)
        for lang in sorted(self.lang_embs_mapping.keys()):
            feature_dict = self.get_features_for_one_language(sim_solver, lang, sorted(self.lang_embs_mapping.keys()), n_closest=n_closest, use_tree=use_tree)
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
        df.to_json(JSON_OUT_PATH)


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
    with open(LANG_PAIRS_map_PATH, "r") as f:
        lang_pairs_map = json.load(f)
    with open(LANG_PAIRS_tree_PATH, "r") as f:
        lang_pairs_tree = json.load(f)
    with open(LANG_PAIRS_ASP_PATH, "rb") as f:
        lang_pairs_asp = pickle.load(f)
    lang_embs = torch.load(LANG_EMBS_PATH)
    with open(LANG_EMBS_MAPPING_PATH, "r") as f:
        lang_embs_mapping = yaml.safe_load(f)
    with open(ISO_LOOKUP_PATH, "r") as f:
        iso_lookup = json.load(f)
    languages_in_text_frontend = get_languages_from_text_frontend()


    return lang_pairs_map, lang_pairs_tree, lang_pairs_asp, lang_embs, lang_embs_mapping, languages_in_text_frontend, iso_lookup




if __name__ == "__main__":

    dc = DatasetCreator()

    # key_error_save_path = "Preprocessing/multilinguality/key_errors_for_languages_from_text_frontend.json"
    # key_error_dict = dc.check_all_languages_in_text_frontend(save_path=key_error_save_path)

    check_features_for_all_languages = False
    if check_features_for_all_languages:
        dc.check_features_for_all_languages()

    #dc.create_json()
    #dc.create_1D_json()
    #dc.create_combined_csv(individual_distances=False)
    #dc.create_combined_csv(individual_distances=True)
    # dc.create_aspf_csv()
    # dc.create_map_csv()
    # dc.create_tree_csv()        
    #dc.create_combined_csv(zero_shot=True)
    # dc.create_aspf_csv(zero_shot=True)
    # dc.create_map_csv(zero_shot=True)
    # dc.create_tree_csv(zero_shot=True)
    # dc.create_random_csv()
        
    dataset_paths = [
        'datasets/dataset_asp_463_with_less_loss_fixed_tree_distance.csv', 
        #'datasets/dataset_asp_463_with_less_loss_fixed_tree_distance_zero_shot.csv', 
        #'datasets/dataset_COMBINED_463_with_less_loss_fixed_tree_distance_average.csv', 
        #'datasets/dataset_COMBINED_463_with_less_loss_fixed_tree_distance_average_individual_dists.csv', 
        #'datasets/dataset_COMBINED_463_with_less_loss_fixed_tree_distance_zero_shot_average.csv', 
        #'datasets/dataset_COMBINED_463_with_less_loss_fixed_tree_distance_zero_shot_average_individual_dists.csv', 
        'datasets/dataset_map_463_with_less_loss_fixed_tree_distance.csv', 
        #'datasets/dataset_map_463_with_less_loss_fixed_tree_distance_zero_shot.csv', 
        'datasets/dataset_random_463_with_less_loss_fixed_tree_distance.csv', 
        'datasets/dataset_tree_463_with_less_loss_fixed_tree_distance.csv', 
        #'datasets/dataset_tree_463_with_less_loss_fixed_tree_distance_zero_shot.csv', 
        ]
    for csv_path in dataset_paths:
        out_path = os.path.join(os.path.dirname(csv_path), f"feature_{os.path.basename(csv_path)}")
        dc.create_feature_csv_from_lookup_csv(csv_path, out_path, single_dim=False)
        out_path = out_path.replace("datasets/", "datasets/single_dim/")
        dc.create_feature_csv_from_lookup_csv(csv_path, out_path, single_dim=True)