import torch
import os

MODEL_PATH = "/home/behringe/hdd_behringe/IMS-Toucan/Models/ToucanTTS_Meta/best.pt"
SAVE_PATH = "/home/behringe/hdd_behringe/IMS-Toucan/Models/LangEmbs/lang_embs_for_15_languages.pt"
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

model = torch.load(MODEL_PATH)

lang_emb_keys = [k for k in model["model"].keys() if "language" in k]
assert len(lang_emb_keys) == 1, "There are multiple candidates for language embedding keys. Please check manually which is correct."

lang_embs = model["model"][lang_emb_keys[0]]
torch.save(lang_embs, SAVE_PATH)