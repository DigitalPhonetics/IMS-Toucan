import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
import sys
sys.path.append("/home/behringe/hdd_behringe/IMS-Toucan")
from Preprocessing.multilinguality.lang_emb_dnn import LangEmbDataset, LangEmbPredictor
from Preprocessing.TextFrontend import load_json_from_path, get_language_id
import datetime
import argparse

# NUM_LANGS = 500
# LANG_EMBS_MAPPING_PATH = f"LangEmbs/mapping_lang_embs_{NUM_LANGS}_langs.yaml"
# with open(LANG_EMBS_MAPPING_PATH, "r") as f:
#     supervised_lang_embs_mapping = yaml.safe_load(f)
# iso_codes_to_ids = load_json_from_path("iso_lookup.json")[-1]
# print(len(iso_codes_to_ids.keys()))
# for sup_lang in supervised_lang_embs_mapping:
#     iso_codes_to_ids.pop(sup_lang, None)
# print(len(iso_codes_to_ids.keys()))

# load dataset including only unseen languages
# in create_lang_emb_dataset, add zero_shot to asp, map, tree -> as in combined
# create_combined_csv, ... with zero_shot=True

device = "cuda" if torch.cuda.is_available() else "cpu"
# create LangEmbDataset with unseen langs
csv_path = "/home/behringe/hdd_behringe/IMS-Toucan/Preprocessing/multilinguality/datasets/dataset_combined_correctsims_463_zero_shot_average.csv"
df = pd.read_csv(csv_path, sep="|")

# load dataset (we only use the features, label is useless)
zero_shot_dataset = LangEmbDataset(dataset_df=df)
data_loader = DataLoader(zero_shot_dataset,
                              batch_size=1,
                              shuffle=False,
                              collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))

# load pretrained language_embedding.weight
load_toucan_model_path = "/home/behringe/hdd_behringe/IMS-Toucan/Preprocessing/multilinguality/toucan_checkpoints/final_model_with_less_loss.pt"
toucan_model = torch.load(load_toucan_model_path)
lang_embs = toucan_model["model"]["encoder.language_embedding.weight"]
lang_embs = lang_embs.to(device)

# load langemb predictor and overwrite unseen languages
langemb_predictor_path = "/home/behringe/hdd_behringe/IMS-Toucan/Preprocessing/multilinguality/checkpoints/240129_154616_10ep/ckpt_10.pt"
LEP_model = LangEmbPredictor(idim=17*5)
LEP_model.load_state_dict(torch.load(langemb_predictor_path))
LEP_model.to(device)
LEP_model.eval()
for data in data_loader:
    x, _, lang_emb_index = data # the label is useless so we ignore it
    prediction = LEP_model(x)
    lang_embs[lang_emb_index] = prediction

# insert into Toucan model
toucan_model["model"]["encoder.language_embedding.weight"] = lang_embs
save_toucan_model_with_modified_langembs_path = "/home/behringe/hdd_behringe/IMS-Toucan/Preprocessing/multilinguality/toucan_checkpoints/models_with_zero_shot_langembs/final_model_with_less_loss_zero_shot_langembs.pt"
torch.save(toucan_model, save_toucan_model_with_modified_langembs_path)