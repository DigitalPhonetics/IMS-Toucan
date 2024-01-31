import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import os
from tqdm import tqdm
import sys
import numpy as np
import pandas as pd
sys.path.append("/home/behringe/hdd_behringe/IMS-Toucan") 
# from Preprocessing.multilinguality.create_lang_emb_dataset import LangEmbDataset
from Preprocessing.TextFrontend import get_language_id

class LangEmbDataset(Dataset):
    def __init__(self,
                 dataset_df,
                 lang_embs_path="LangEmbs/final_model_with_less_loss.pt",
                 add_noise=False,
                 noise_std=0.01):
        self.dataset_df = dataset_df
        self.add_noise = add_noise
        self.noise_std = noise_std if add_noise else None
         # for combined feats, df has 5 features per closest lang + 1 target lang column
        if "average_dist_0" in self.dataset_df.columns or "euclidean_dist_0" in self.dataset_df.columns:
            self.n_closest = len(self.dataset_df.columns) // 5
            self.distance_type = "average" if  "average_dist_0" in self.dataset_df.columns else "euclidean"
        # else, df has 2 features per closest lang + 1 target lang column
        else:
            self.n_closest = len(self.dataset_df.columns) // 2
            if "map_dist_0" in self.dataset_df.columns:
                self.distance_type = "map"
            elif "tree_dist_0" in self.dataset_df.columns:
                self.distance_type = "tree"
            elif "asp_dist_0" in self.dataset_df.columns:
                self.distance_type = "asp"
            else:
                self.distance_type = "fixed" # for random dataset
        

        self.language_embeddings = torch.load(lang_embs_path)

    def __len__(self):
        return len(self.dataset_df)
    
    def __getitem__(self, idx):
        """Return tuple of features and label (as tensors), 
        and the language_embedding id (for zero-shot prediction to replace lang_emb at correct index)."""
        features = self.dataset_df.iloc[idx, :]
        target_lang_code = features["target_lang"]
        target_lang_emb_index = get_language_id(target_lang_code)
        if target_lang_emb_index is None:
            print(f"target_lang_code: {target_lang_code}")
        dist_plus_lang_emb_tensors = []
        y = self.language_embeddings[target_lang_emb_index.item()]
        if self.add_noise:
            noise = torch.normal(mean=0, std=0.01, size=y.size())
            y += noise
        for i in range(self.n_closest):
            dist = torch.tensor([features[f"{self.distance_type}_dist_{i}"]], dtype=torch.float32)
            lang = features[f"closest_lang_{i}"]
            lang_emb = self.language_embeddings[get_language_id(lang).item()]
            if self.add_noise:
                noise = torch.normal(mean=0, std=self.noise_std, size=lang_emb.size())
                lang_emb += noise
            dist_plus_lang_emb_tensors.append(torch.cat((dist, lang_emb)))
        feature_tensor = torch.cat(dist_plus_lang_emb_tensors)
        return feature_tensor, y, target_lang_emb_index


class LangEmbPredictor(torch.nn.Module):
    def __init__(self, idim, hdim=16, odim=16, n_layers=2, dropout_rate=3, n_closest=5):
        super().__init__()
        self.linear1 = torch.nn.Linear(idim, hdim)
        self.linear2 = torch.nn.Linear(hdim, odim)
        self.leaky_relu = torch.nn.LeakyReLU()
        self.layers = torch.nn.Sequential(self.linear1, 
                                          self.leaky_relu,
                                          self.linear2)

    def forward(self, xs):
        xs = self.layers(xs)
        return xs
    
def train(model: LangEmbPredictor, 
          train_set: LangEmbDataset, 
          device, 
          checkpoint_dir,
          test_set: LangEmbDataset = None, 
          lr=0.001,
          batch_size=4,
          n_epochs: int = 10,
          save_ckpt_every=10):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.to(device)    
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
    if test_set:
        test_loader = DataLoader(test_set,
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
    for epoch in tqdm(range(n_epochs), total=n_epochs, desc="Epoch"):
        model.train()
        running_loss = 0.
        for _, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            x, y, _ = data
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)

        if test_set:
            model.eval()
            running_val_loss = 0.
            with torch.inference_mode():
                for _, data in tqdm(enumerate(test_loader), total=len(test_loader)):
                    val_x, val_y, _ = data
                    val_outputs = model(val_x)
                    val_loss = loss_fn(val_outputs, val_y)
                    running_val_loss += val_loss.item()
            avg_val_loss = running_val_loss / len(test_loader)
        print(f"Epoch {epoch+1} | Train loss: {avg_train_loss} | Val loss: {avg_val_loss}")
        if epoch > 0 and (epoch+1) % save_ckpt_every == 0:
            model_path = os.path.join(checkpoint_dir, f"ckpt_{epoch+1}.pt")
            torch.save(model.state_dict(), model_path)
    if test_set:
        print(f"Final train loss: {avg_train_loss} | Final val loss: {avg_val_loss}")
        return avg_train_loss, avg_val_loss        
    else:
        print(f"Train loss: {avg_train_loss}")
        return avg_train_loss
