import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import os
import numpy as np
import pandas as pd
import json
import sys
sys.path.append("/home/behringe/hdd_behringe/IMS-Toucan")
from Preprocessing.multilinguality.lang_emb_dnn import LangEmbDataset, LangEmbPredictor
import argparse

def main(inference_mode, min_n_langs=5, max_n_langs=30, threshold_percentile=95):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load pretrained language_embedding.weight
    load_toucan_model_path = "toucan_checkpoints/final_model_with_less_loss_fixed_tree_distance.pt"
    toucan_model = torch.load(load_toucan_model_path)
    lang_embs = toucan_model["model"]["encoder.language_embedding.weight"]
    lang_embs = lang_embs.to(device)

    save_toucan_model_with_modified_langembs_path = f"/home/behringe/hdd_behringe/IMS-Toucan/Preprocessing/multilinguality/toucan_checkpoints/models_with_zero_shot_langembs/final_model_with_less_loss_fixed_tree_distance_zero_shot_langembs_learned_min{min_n_langs}_max{max_n_langs}_thresholdpercentile{threshold_percentile}_{inference_mode}_no_illegal_langs.pt"

    # create LangEmbDataset with unseen langs
    csv_path = "/home/behringe/hdd_behringe/IMS-Toucan/Preprocessing/multilinguality/datasets/dataset_learned_dist_463_with_less_loss_fixed_tree_distance_top50_zero_shot_no_illegal_langs.csv"
    df = pd.read_csv(csv_path, sep="|")

    if inference_mode == "average":
        if "average_dist_0" in df.columns and "map_dist_0" in df.columns or "euclidean_dist_0" in df.columns and "map_dist_0" in df.columns:
            n_closest = len(df.columns) // 5
            distance_type = "average" if "average_dist_0" in df.columns else "euclidean"
        # else, df has 2 features per closest lang + 1 target lang column
        else:
            n_closest = len(df.columns) // 2
            if "average_dist_0" in df.columns:
                distance_type = "average"
            elif "euclidean_dist_0" in df.columns:
                distance_type = "euclidean"
            elif "map_dist_0" in df.columns:
                distance_type = "map"
            elif "tree_dist_0" in df.columns:
                distance_type = "tree"
            elif "asp_dist_0" in df.columns:
                distance_type = "asp"
            elif "learned_dist_0" in df.columns:
                distance_type = "learned"
            else:
                distance_type = "fixed" # for random dataset

        closest_lang_columns = [f"closest_lang_{i}" for i in range(n_closest)]
        closest_dist_columns = [f"{distance_type}_dist_{i}" for i in range(n_closest)]
        closest_lang_columns = closest_lang_columns[:max_n_langs]
        closest_dist_columns = closest_dist_columns[:max_n_langs]
        # get threshold: median distance of least closest language
        threshold = np.percentile(df[closest_dist_columns[-1]], threshold_percentile)
        # set minimum number of languages

        for row in df.itertuples():
            avg_emb = torch.zeros([16]).to(device)
            dists = [getattr(row, d) for i, d in enumerate(closest_dist_columns) if i < min_n_langs or getattr(row, d) < threshold]
            langs = [getattr(row, l) for l in closest_lang_columns[:len(dists)]]

            for lang in langs:
                lang_emb = lang_embs[iso_lookup[-1][str(lang)]]
                avg_emb += lang_emb
            normalization_factor = len(langs)
            avg_emb /= normalization_factor # normalize
            lang_embs[iso_lookup[-1][str(row.target_lang)]] = avg_emb


    elif inference_mode == "mlp":
        if "individual_dists" in csv_path:
            use_individual_distances = True
            idim = 19*5
        else:
            use_individual_distances = False
            idim = 17*5

        # load dataset (we only use the features, label is useless)
        zero_shot_dataset = LangEmbDataset(dataset_df=df, use_individual_distances=use_individual_distances)
        data_loader = DataLoader(zero_shot_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))


        # load langemb predictor and overwrite unseen languages
        langemb_predictor_path = "checkpoints/240207_204631_30ep/ckpt_ep30.pt"
        LEP_model = LangEmbPredictor(idim)
        LEP_model.load_state_dict(torch.load(langemb_predictor_path))
        LEP_model.to(device)
        LEP_model.eval()
        faulty_df_indices = []
        with torch.inference_mode():
            for data in data_loader:
                try:
                    x, _, lang_emb_index = data # the label is useless so we ignore it
                    prediction = LEP_model(x)
                    lang_embs[lang_emb_index] = prediction
                except:
                    # in case of error, only the dataframe index is returned which cause the error
                    faulty_df_idx = data
                    faulty_df_indices.append(faulty_df_idx)

        print(f"Failed to predict language embedding for the following indices of the used DataFrame: {faulty_df_indices}")
    else:
        raise ValueError

    # insert into Toucan model
    toucan_model["model"]["encoder.language_embedding.weight"] = lang_embs
    torch.save(toucan_model, save_toucan_model_with_modified_langembs_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_mode", choices=["mlp", "average"], help="which method to use to generate zero-shot embeddings")
    parser.add_argument("--min_n_langs", type=int, default=5, help="minimum amount of languages used for averaging")
    parser.add_argument("--max_n_langs", type=int, default=30, help="maximum amount of languages used for averaging")
    parser.add_argument("--threshold_percentile", type=int, default=95, help="percentile of the furthest used languages \
                        used as cutoff threshold (no langs >= the threshold are used for averagin)")    
    args = parser.parse_args() 
    ISO_LOOKUP_PATH = "iso_lookup.json"
    with open(ISO_LOOKUP_PATH, "r") as f:
        iso_lookup = json.load(f)    
    main(inference_mode=args.inference_mode, 
         min_n_langs=args.min_n_langs,
         max_n_langs=args.max_n_langs,
         threshold_percentile=args.threshold_percentile)