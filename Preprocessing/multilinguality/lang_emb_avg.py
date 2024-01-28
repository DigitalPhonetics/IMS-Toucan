import torch
import sys
import os
import numpy as np
import pandas as pd
sys.path.append("/home/behringe/hdd_behringe/IMS-Toucan")
from Preprocessing.TextFrontend import get_language_id

# csv_path = "datasets/dataset_COMBINED_500_with_less_loss_average.csv"
csv_path = "datasets/dataset_COMBINED_correct_sims_500_with_less_loss_average.csv"
#csv_path = "datasets/dataset_asp_500_with_less_loss.csv"
#csv_path = "datasets/dataset_map_500_with_less_loss.csv"
#csv_path = "datasets/dataset_tree_500_with_less_loss.csv"
dataset_df = pd.read_csv(csv_path, sep="|")

lang_embs_path = "LangEmbs/final_model_with_less_loss.pt"
language_embeddings = torch.load(lang_embs_path)

loss_fn = torch.nn.MSELoss()
running_loss = 0.

# for combined feats, df has 5 features per closest lang + 1 target lang column
if "average_dist_0" in dataset_df.columns or "euclidean_dist_0" in dataset_df.columns:
    n_closest = len(dataset_df.columns) // 5
# else, df has 2 features per closest lang + 1 target lang column
else:
    n_closest = len(dataset_df.columns) // 2
assert n_closest == 5

for row in dataset_df.itertuples():
    target_lang = row.target_lang
    y = language_embeddings[get_language_id(target_lang).item()]
    avg_emb = torch.zeros([16])
    langs = [row.closest_lang_0, row.closest_lang_1, row.closest_lang_2, row.closest_lang_3, row.closest_lang_4]
    for lang in langs:
        lang_emb = language_embeddings[get_language_id(lang).item()]
        avg_emb += lang_emb
    avg_emb /= 5 # take average
    current_loss = loss_fn(avg_emb, y).item()
    running_loss += current_loss
avg_loss = running_loss / len(dataset_df)

print(f"csv_path: {csv_path}")
print(avg_loss)