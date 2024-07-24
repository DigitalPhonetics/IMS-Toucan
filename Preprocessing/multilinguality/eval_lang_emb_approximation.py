import argparse
import os

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 7
import matplotlib.pyplot as plt
from Utility.utils import load_json_from_path
from Utility.storage_config import MODELS_DIR

def compute_loss_for_approximated_embeddings(csv_path, iso_lookup, language_embeddings, weighted_avg=False, min_n_langs=5, max_n_langs=30, threshold_percentile=95, loss_fn="MSE"):
    df = pd.read_csv(csv_path, sep="|")

    if loss_fn == "L1":
        loss_fn = torch.nn.L1Loss()
    else:
        loss_fn = torch.nn.MSELoss()

    features_per_closest_lang = 2
    # for combined, df has up to 5 features (if containing individual distances) per closest lang + 1 target lang column
    if "combined_dist_0" in df.columns: 
        if "map_dist_0" in df.columns:
            features_per_closest_lang += 1
        if "asp_dist_0" in df.columns:
            features_per_closest_lang += 1
        if "tree_dist_0" in df.columns:
            features_per_closest_lang += 1
        n_closest = len(df.columns) // features_per_closest_lang
        distance_type = "combined"
    # else, df has 2 features per closest lang + 1 target lang column        
    else:
        n_closest = len(df.columns) // features_per_closest_lang
        if "map_dist_0" in df.columns:
            distance_type = "map"
        elif "tree_dist_0" in df.columns:
            distance_type = "tree"
        elif "asp_dist_0" in df.columns:
            distance_type = "asp"
        elif "learned_dist_0" in df.columns:
            distance_type = "learned"
        elif "oracle_dist_0" in df.columns:
            distance_type = "oracle"
        else:
            distance_type = "random"

    closest_lang_columns = [f"closest_lang_{i}" for i in range(n_closest)]
    closest_dist_columns = [f"{distance_type}_dist_{i}" for i in range(n_closest)]
    closest_lang_columns = closest_lang_columns[:max_n_langs]
    closest_dist_columns = closest_dist_columns[:max_n_langs]

    threshold = np.percentile(df[closest_dist_columns[-1]], threshold_percentile)
    print(f"threshold: {threshold}")
    all_losses = []

    for row in df.itertuples():
        try:
            y = language_embeddings[iso_lookup[-1][row.target_lang]]
        except KeyError:
            print(f"KeyError: Unable to retrieve language embedding for {row.target_lang}")
            continue
        avg_emb = torch.zeros([16])
        dists = [getattr(row, d) for i, d in enumerate(closest_dist_columns) if i < min_n_langs or getattr(row, d) < threshold]
        langs = [getattr(row, l) for l in closest_lang_columns[:len(dists)]]

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
        all_losses.append(current_loss)

    return all_losses


if __name__ == "__main__":
    default_model_path = os.path.join("../..", MODELS_DIR, "ToucanTTS_Meta", "best.pt")  # MODELS_DIR must be absolute path, the relative path will fail at this location
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=default_model_path, help="model path that should be used for creating oracle lang emb distance cache")
    parser.add_argument("--min_n_langs", type=int, default=5, help="minimum amount of languages used for averaging")
    parser.add_argument("--max_n_langs", type=int, default=30, help="maximum amount of languages used for averaging")
    parser.add_argument("--threshold_percentile", type=int, default=95, help="percentile of the furthest used languages \
                        used as cutoff threshold (no langs >= the threshold are used for averagin)")
    parser.add_argument("--loss_fn", choices=["MSE", "L1"], type=str, default="MSE", help="loss function used")
    args = parser.parse_args()
    csv_paths = [
        "distance_datasets/dataset_map_top30_furthest.csv",
        "distance_datasets/dataset_random_top30.csv",
        "distance_datasets/dataset_asp_top30.csv",
        "distance_datasets/dataset_tree_top30.csv",
        "distance_datasets/dataset_map_top30.csv",
        "distance_datasets/dataset_combined_top30_indiv-dists.csv",
        "distance_datasets/dataset_learned_top30.csv",
        "distance_datasets/dataset_oracle_top30.csv",
    ]
    weighted = [False]
    lang_embs = torch.load(args.model_path)["model"]["encoder.language_embedding.weight"]
    lang_embs.requires_grad_(False)
    iso_lookup = load_json_from_path("iso_lookup.json")
    losses_of_multiple_datasets = []
    OUT_DIR = "plots"
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.15022, 3.15022*(2/3)), constrained_layout=True)
    plt.ylabel(args.loss_fn)
    for i, csv_path in enumerate(csv_paths):
        print(f"csv_path: {os.path.basename(csv_path)}")
        for condition in weighted:
            losses = compute_loss_for_approximated_embeddings(csv_path, 
                                                         iso_lookup, 
                                                         lang_embs, 
                                                         condition, 
                                                         min_n_langs=args.min_n_langs, 
                                                         max_n_langs=args.max_n_langs,
                                                         threshold_percentile=args.threshold_percentile,
                                                         loss_fn=args.loss_fn)
            print(f"weighted average: {condition} | mean loss: {np.mean(losses)}")
            losses_of_multiple_datasets.append(losses)

    bp_dict = ax.boxplot(losses_of_multiple_datasets, 
                         labels = [
                             "map furthest",
                             "random", 
                             "inv. ASP", 
                             "tree", 
                             "map", 
                             "avg", 
                             "meta-learned", 
                             "oracle", 
                             ], 
                         patch_artist=True,
                         boxprops=dict(facecolor = "lightblue", 
                                       ),
                        showfliers=False,
                        widths=0.45
                        )

    # major ticks every 0.1, minor ticks every 0.05, between 0.0 and 0.6
    major_ticks = np.arange(0, 0.6, 0.1)
    minor_ticks = np.arange(0, 0.6, 0.05)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    # horizontal grid lines for minor and major ticks
    ax.grid(which='both', linestyle='-', color='lightgray', linewidth=0.3, axis='y')
    ax.set_aspect(4.5)
    plt.title(f"min. {args.min_n_langs} kNN, max. {args.max_n_langs}\nthreshold: {args.threshold_percentile}th-percentile distance of {args.max_n_langs}th-closest language")
    plt.xticks(rotation=45)

    plt.savefig(os.path.join(OUT_DIR, "example_boxplot_release.pdf"), bbox_inches='tight')
