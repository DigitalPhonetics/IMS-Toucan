import torch
import sys
import os
import numpy as np
import pandas as pd
sys.path.append("/home/behringe/hdd_behringe/IMS-Toucan")
from Preprocessing.TextFrontend import get_language_id


def compute_mse_for_averaged_embeddings(csv_path, language_embeddings, weighted_avg=False):
    dataset_df = pd.read_csv(csv_path, sep="|")

    loss_fn = torch.nn.MSELoss()
    running_loss = 0.

    # for combined feats, df has 5 features per closest lang + 1 target lang column
    if "average_dist_0" in dataset_df.columns or "euclidean_dist_0" in dataset_df.columns:
        n_closest = len(dataset_df.columns) // 5
        distance_type = "average" if "average_dist_0" in dataset_df.columns else "euclidean"
    # else, df has 2 features per closest lang + 1 target lang column
    else:
        n_closest = len(dataset_df.columns) // 2
        if "map_dist_0" in dataset_df.columns:
            distance_type = "map"
        elif "tree_dist_0" in dataset_df.columns:
            distance_type = "tree"
        elif "asp_dist_0" in dataset_df.columns:
            distance_type = "asp"
        else:
            distance_type = "fixed" # for random dataset
    assert n_closest == 5

    for row in dataset_df.itertuples():
        y = language_embeddings[get_language_id(row.target_lang).item()]
        avg_emb = torch.zeros([16])
        langs = [row.closest_lang_0, row.closest_lang_1, row.closest_lang_2, row.closest_lang_3, row.closest_lang_4]
        if distance_type == "average":
            dists = [row.average_dist_0, row.average_dist_1, row.average_dist_2, row.average_dist_3, row.average_dist_4]
        elif distance_type == "asp":
            dists = [row.asp_dist_0, row.asp_dist_1, row.asp_dist_2, row.asp_dist_3, row.asp_dist_4]
        elif distance_type == "map":
            dists = [row.map_dist_0, row.map_dist_1, row.map_dist_2, row.map_dist_3, row.map_dist_4]
        elif distance_type == "tree":
            dists = [row.tree_dist_0, row.tree_dist_1, row.tree_dist_2, row.tree_dist_3, row.tree_dist_4]
        else:
            dists = [row.fixed_dist_0, row.fixed_dist_1, row.fixed_dist_2, row.fixed_dist_3, row.fixed_dist_4]

        if weighted_avg:
            for lang, dist in zip(langs, dists):
                lang_emb = language_embeddings[get_language_id(lang).item()]
                avg_emb += lang_emb * dist
            normalization_factor = sum(dists)
        else:
            for lang in langs:
                lang_emb = language_embeddings[get_language_id(lang).item()]
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

    ]
    weighted = [True, False]
    #lang_embs_path = "LangEmbs/final_model_with_less_loss.pt"
    lang_embs_path = "LangEmbs/final_model_with_less_loss_fixed_tree_distance.pt"
    language_embeddings = torch.load(lang_embs_path)

    for csv_path in csv_paths:
        print(f"csv_path: {csv_path}")
        for condition in weighted:
            avg_loss = compute_mse_for_averaged_embeddings(csv_path, language_embeddings, condition)
            print(f"weighted average: {condition} | avg loss: {avg_loss}")
