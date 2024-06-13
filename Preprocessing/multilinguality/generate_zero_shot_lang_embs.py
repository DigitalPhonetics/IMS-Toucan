import torch
import numpy as np
import pandas as pd
import json
import argparse
from tqdm import tqdm
import os
from Utility.storage_config import MODELS_DIR

def approximate_and_inject_language_embeddings(model_path, df, iso_lookup, min_n_langs=5, max_n_langs=25, threshold_percentile=50):
    # load pretrained language_embeddings
    model = torch.load(model_path, map_location="cpu")
    lang_embs = model["model"]["encoder.language_embedding.weight"]

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
        else:
            distance_type = "random"

    # get relevant columns
    closest_lang_columns = [f"closest_lang_{i}" for i in range(n_closest)]
    closest_dist_columns = [f"{distance_type}_dist_{i}" for i in range(n_closest)]
    closest_lang_columns = closest_lang_columns[:max_n_langs]
    closest_dist_columns = closest_dist_columns[:max_n_langs]
    assert df[closest_dist_columns[-1]].isna().sum().sum() == 0

    # get threshold based on distance of a certain percentile of the furthest language across all samples
    threshold = np.percentile(df[closest_dist_columns[-1]], threshold_percentile)
    print(f"threshold: {threshold:.4f}")
    for row in tqdm(df.itertuples(), total=df.shape[0], desc="Approximating language embeddings"):
        avg_emb = torch.zeros([16])
        dists = [getattr(row, d) for i, d in enumerate(closest_dist_columns) if i < min_n_langs or getattr(row, d) < threshold]
        langs = [getattr(row, l) for l in closest_lang_columns[:len(dists)]]

        for lang in langs:
            lang_emb = lang_embs[iso_lookup[-1][str(lang)]]
            avg_emb += lang_emb
        avg_emb /= len(langs) # normalize
        lang_embs[iso_lookup[-1][str(row.target_lang)]] = avg_emb

    # inject language embeddings into Toucan model and save
    model["model"]["encoder.language_embedding.weight"] = lang_embs
    modified_model_path = model_path.split(".")[0] + "_zeroshot_lang_embs.pt"
    torch.save(model, modified_model_path)
    print(f"Replaced unsupervised language embeddings with zero-shot approximations.\nSaved modified model to {modified_model_path}")


if __name__ == "__main__":
    default_model_path = os.path.join(MODELS_DIR, "ToucanTTS_Meta", "best.pt") # MODELS_DIR must be absolute path, the relative path will fail at this location
    default_csv_path = "distance_datasets/dataset_learned_top30.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=default_model_path, help="path of the model for which the language embeddings should be modified")
    parser.add_argument("--dataset_path", type=str, default=default_csv_path, help="path to distance dataset CSV")
    parser.add_argument("--min_n_langs", type=int, default=5, help="minimum amount of languages used for averaging")
    parser.add_argument("--max_n_langs", type=int, default=25, help="maximum amount of languages used for averaging")
    parser.add_argument("--threshold_percentile", type=int, default=50, help="percentile of the furthest used languages \
                        used as cutoff threshold (no langs >= the threshold are used for averaging)")
    args = parser.parse_args() 
    ISO_LOOKUP_PATH = "iso_lookup.json"
    with open(ISO_LOOKUP_PATH, "r") as f:
        iso_lookup = json.load(f) # iso_lookup[-1] = iso2id mapping
    # load language distance dataset
    distance_df = pd.read_csv(args.dataset_path, sep="|")
    approximate_and_inject_language_embeddings(model_path=args.model_path,
                                  df=distance_df,
                                  iso_lookup=iso_lookup,
                                  min_n_langs=args.min_n_langs,
                                  max_n_langs=args.max_n_langs,
                                  threshold_percentile=args.threshold_percentile)
    