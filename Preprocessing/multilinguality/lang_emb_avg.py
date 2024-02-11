import torch
import sys
import os
import numpy as np
import pandas as pd
sys.path.append("/home/behringe/hdd_behringe/IMS-Toucan")
import json



def compute_mse_for_averaged_embeddings(csv_path, iso_lookup, language_embeddings, weighted_avg=False):
    dataset_df = pd.read_csv(csv_path, sep="|")

    loss_fn = torch.nn.MSELoss()
    running_loss = 0.

    # for combined feats, df has 5 features per closest lang + 1 target lang column
    if "average_dist_0" in dataset_df.columns and "map_dist_0" in dataset_df.columns or "euclidean_dist_0" in dataset_df.columns and "map_dist_0" in dataset_df.columns:
        n_closest = len(dataset_df.columns) // 5
        distance_type = "average" if "average_dist_0" in dataset_df.columns else "euclidean"
    # else, df has 2 features per closest lang + 1 target lang column
    else:
        n_closest = len(dataset_df.columns) // 2
        if "average_dist_0" in dataset_df.columns:
            distance_type = "average"
        elif "euclidean_dist_0" in dataset_df.columns:
            distance_type = "euclidean"
        elif "map_dist_0" in dataset_df.columns:
            distance_type = "map"
        elif "tree_dist_0" in dataset_df.columns:
            distance_type = "tree"
        elif "asp_dist_0" in dataset_df.columns:
            distance_type = "asp"
        else:
            distance_type = "fixed" # for random dataset

    closest_lang_columns = [f"closest_lang_{i}" for i in range(n_closest)]
    closest_dist_columns = [f"{distance_type}_dist_{i}" for i in range(n_closest)]

    for row in dataset_df.itertuples():
        y = language_embeddings[iso_lookup[-1][row.target_lang]]
        avg_emb = torch.zeros([16])
        langs = [getattr(row, l) for l in closest_lang_columns]
        dists = [getattr(row, d) for d in closest_dist_columns]

        if weighted_avg:
            for lang, dist in zip(langs, dists):
                lang_emb = language_embeddings[iso_lookup[-1][lang]]
                avg_emb += lang_emb * dist
            normalization_factor = sum(dists)
        else:
            for lang in langs:
                lang_emb = language_embeddings[iso_lookup[-1][lang]]
                avg_emb += lang_emb
            normalization_factor = len(langs)
        avg_emb /= normalization_factor # normalize
        current_loss = loss_fn(avg_emb, y).item()
        running_loss += current_loss
        # print(y)
        # print(avg_emb)
        # l1loss = torch.nn.L1Loss(reduction="none")
        # print(l1loss(y, avg_emb))
        # print(running_loss)
    avg_loss = running_loss / len(dataset_df)
    return avg_loss

if __name__ == "__main__":
    csv_paths = [
        #"datasets/OLD/dataset_COMBINED_500_with_less_loss_average.csv",
        #"datasets/OLD/dataset_COMBINED_correct_sims_500_with_less_loss_average.csv",
        #"datasets/OLD/dataset_random_463_with_less_loss.csv",
        #"datasets/OLD/dataset_asp_463_with_less_loss.csv",
        #"datasets/OLD/dataset_map_463_with_less_loss.csv",
        #"datasets/dataset_tree_463_with_less_loss.csv",
        #"datasets/dataset_asp_463_with_less_loss_fixed_tree_distance.csv",
        #"datasets/dataset_COMBINED_463_with_less_loss_fixed_tree_distance_average.csv",
        #"datasets/dataset_map_463_with_less_loss_fixed_tree_distance.csv",
        #"datasets/dataset_tree_463_with_less_loss_fixed_tree_distance.csv",
        #"datasets/dataset_random_463_with_less_loss_fixed_tree_distance.csv",

        # 10-100 kNN, with individual dists
        #"datasets/dataset_COMBINED_463_with_less_loss_fixed_tree_distance_top10_average_individual_dists.csv",
        #"datasets/dataset_COMBINED_463_with_less_loss_fixed_tree_distance_top20_average_individual_dists.csv",
        #"datasets/dataset_COMBINED_463_with_less_loss_fixed_tree_distance_top25_average_individual_dists.csv",
        #"datasets/dataset_COMBINED_463_with_less_loss_fixed_tree_distance_top30_average_individual_dists.csv",
        #"datasets/dataset_COMBINED_463_with_less_loss_fixed_tree_distance_top40_average_individual_dists.csv",        
        #"datasets/dataset_COMBINED_463_with_less_loss_fixed_tree_distance_top50_average_individual_dists.csv",
        #"datasets/dataset_COMBINED_463_with_less_loss_fixed_tree_distance_top100_average_individual_dists.csv"

        # RANDOM
        #"datasets/dataset_random_463_with_less_loss_fixed_tree_distance_random20.csv",
        #"datasets/dataset_random_463_with_less_loss_fixed_tree_distance_random25.csv",
        #"datasets/dataset_random_463_with_less_loss_fixed_tree_distance_random30.csv",

        # LEARNED WEIGHTS (with/without individual dists)

        "datasets/dataset_COMBINED_463_with_less_loss_fixed_tree_distance_average_individual_dists_learned_weights.csv",
        "datasets/dataset_COMBINED_463_with_less_loss_fixed_tree_distance_top20_average_individual_dists_learned_weights.csv",
        "datasets/dataset_COMBINED_463_with_less_loss_fixed_tree_distance_top25_average_individual_dists_learned_weights.csv",
        "datasets/dataset_COMBINED_463_with_less_loss_fixed_tree_distance_top30_average_individual_dists_learned_weights.csv",
        #"datasets/dataset_COMBINED_463_with_less_loss_fixed_tree_distance_average_learned_weights.csv",        
        # "datasets/dataset_COMBINED_463_with_less_loss_fixed_tree_distance_top20_average_learned_weights.csv",
        # "datasets/dataset_COMBINED_463_with_less_loss_fixed_tree_distance_top25_average_learned_weights.csv",
        # "datasets/dataset_COMBINED_463_with_less_loss_fixed_tree_distance_top30_average_learned_weights.csv",

    ]
    weighted = [True, False]
    lang_embs_path = "LangEmbs/final_model_with_less_loss_fixed_tree_distance.pt"
    language_embeddings = torch.load(lang_embs_path)

    ISO_LOOKUP_PATH = "iso_lookup.json"
    with open(ISO_LOOKUP_PATH, "r") as f:
        iso_lookup = json.load(f)

    for csv_path in csv_paths:
        print(f"csv_path: {csv_path}")
        for condition in weighted:
            avg_loss = compute_mse_for_averaged_embeddings(csv_path, iso_lookup, language_embeddings, condition)
            print(f"weighted average: {condition} | avg loss: {avg_loss}")
