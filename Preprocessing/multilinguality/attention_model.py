import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import os
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import datetime
import argparse
import sys
sys.path.append("/media/hdd/behringe/IMS-Toucan")
from Preprocessing.multilinguality import asp
from Preprocessing.multilinguality.SimilaritySolver import SimilaritySolver
from Preprocessing.TextFrontend import get_language_id
from time import time

class ScoringModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scoring_function = torch.nn.Sequential(torch.nn.Linear(3, 3),
                                        torch.nn.Tanh(),
                                        torch.nn.Linear(3, 3),
                                        torch.nn.Tanh(),
                                        torch.nn.Linear(3, 1),
                                        torch.nn.Softmax(dim=1))
    
    def forward(self, x):
        return self.scoring_function(x)


class DistanceDataset(Dataset):
    def __init__(self,
                 lang_only_df,
                 dist_df,
                 partial_lang_embs_df,
                 full_lang_embs_df,
                 exclude_ids_df):
        assert len(lang_only_df) == len(dist_df) == len(partial_lang_embs_df) == len(exclude_ids_df)
        self.lang_only_df = lang_only_df
        self.dist_df = dist_df
        self.all_lang_codes = list(lang_only_df.iloc[:, 0])
        self.supervised_language_emb_df = partial_lang_embs_df
        self.full_supervised_lang_emb_df = full_lang_embs_df
        self.exclude_ids_df = exclude_ids_df
        self.sim_solver = SimilaritySolver()
        
    def __len__(self):
        return len(self.lang_only_df)

    def __getitem__(self, idx):
        distances = self.dist_df.iloc[idx, :]
        # TODO: make distance length check nicer
        view_dim_0 = len(self.dist_df)
        if len(distances) / 3 != view_dim_0:
            view_dim_0 -= 1
        assert len(distances) / 3 == view_dim_0, f"{len(distances)/3} != {view_dim_0}"

        distances = torch.tensor(distances.values, dtype=torch.float32)
        distances = distances.view(view_dim_0, 3)
        exclude_indices = self.exclude_ids_df.iloc[idx, :]
        lang_embs = self.supervised_language_emb_df.drop(exclude_indices)
        lang_embs = torch.tensor(lang_embs.values, dtype=torch.float32)
        y = torch.tensor(self.full_supervised_lang_emb_df.iloc[exclude_indices[0], :], dtype=torch.float32)
        return distances, lang_embs, y

def train(model: ScoringModel, 
          train_set,
          device,
          checkpoint_dir,
          test_set=None,
          lr=0.001,
          batch_size=16,
          n_epochs=10,
          save_ckpt_every=10,
          model_id=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_id_suffix = f"_model{model_id}" if model_id else ""
    model.to(device)
    loss_fn = torch.nn.L1Loss()
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
        for step, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            # retrieve distance_tensor, target_lang_emb, and all lang_codes
            distances, lang_embs_without_target, y = data
            optimizer.zero_grad()
            scores = []
            if distances.shape[1] != lang_embs_without_target.shape[1]:
                distances = distances[:, :-1, :]
            assert distances.shape[1] == lang_embs_without_target.shape[1], f"{distances.shape[1]} != {lang_embs_without_target.shape[1]}"
            for i in range(distances.shape[1]):
                score = model(distances[:, i, :])
                scores.append(score)
            scores = torch.stack(scores, dim=1)
            weighted_lang_embs_without_target = scores * lang_embs_without_target
            predicted_target_lang_emb = weighted_lang_embs_without_target.sum(dim=1) / distances.shape[1]
            print(f"Pred:\n{predicted_target_lang_emb}")
            print(f"GT:\n{y}")
            loss = loss_fn(predicted_target_lang_emb, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f"Step {step+1} | Train loss: {loss.item()}")
        avg_train_loss = running_loss / len(train_loader)

        if test_set:
            model.eval()
            running_val_loss = 0.
            with torch.inference_mode():
                for _, data in tqdm(enumerate(test_loader), total=len(test_loader)):
                    val_distances, val_lang_embs_without_target, val_y = data
                    val_scores = []
                    if val_distances.shape[1] != val_lang_embs_without_target.shape[1]:
                        val_distances = val_distances[:, :-1, :]
                    assert val_distances.shape[1] == val_lang_embs_without_target.shape[1], f"{val_distances.shape[1]} != {val_lang_embs_without_target.shape[1]}"                    
                    for i in range(val_distances.shape[1]):
                        val_score = model(val_distances[:, i, :])
                        val_scores.append(val_score)
                    val_scores = torch.stack(val_scores, dim=1)
                    val_weighted_lang_embs = val_scores * val_lang_embs_without_target
                    val_predicted_target_lang_emb = val_weighted_lang_embs.sum(dim=1)
                    val_loss = loss_fn(val_predicted_target_lang_emb, val_y)
                    running_val_loss += val_loss.item()
            avg_val_loss = running_val_loss / len(test_loader)
            print(f"Epoch {epoch+1} | Train loss: {avg_train_loss} | Val loss: {avg_val_loss}")
        else:
            print(f"Epoch {epoch+1} | Train loss: {avg_train_loss}")
        if epoch > 0 and (epoch+1) % save_ckpt_every == 0:
            model_path = os.path.join(checkpoint_dir, f"ckpt{model_id_suffix}_ep{epoch+1}.pt")
            torch.save(model.state_dict(), model_path)
    if test_set:
        print(f"Final train loss: {avg_train_loss} | Final val loss: {avg_val_loss}")
        return avg_train_loss, avg_val_loss        
    else:
        print(f"Train loss: {avg_train_loss}")
        return avg_train_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", help="path to dataset csv")
    parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size for each model step")
    parser.add_argument("--checkpoint_dir", help="directory for saving checkpoints")
    args = parser.parse_args()

    log_dir = "logs"
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else f"checkpoints/{timestamp}_att_{args.batch_size}_{args.n_epochs}ep"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # df = pd.read_csv(args.csv_path, sep="|")
    lang_only_df = pd.read_csv("datasets/only_lang_codes.csv", sep="|")
    lang_embs_df = pd.read_csv("LangEmbs/all_supervised_lang_embs.csv", sep="|")
    dist_df = pd.read_csv("datasets/all_dists.csv", sep="|")
    exclude_ids_df = pd.read_csv("datasets/exclude_lang_indices.csv", sep="|")
    model = ScoringModel()
    print("Initializing datasets.")
    train_set = DistanceDataset(lang_only_df=lang_only_df[:410],
                                dist_df=dist_df[:410].iloc[:,:410*3],
                                partial_lang_embs_df=lang_embs_df[:410],
                                full_lang_embs_df=lang_embs_df,
                                exclude_ids_df=exclude_ids_df[:410])
    test_set = DistanceDataset(lang_only_df=lang_only_df[410:],
                               dist_df=dist_df[410:].iloc[:,410*3:],
                               partial_lang_embs_df=lang_embs_df[410:],
                               full_lang_embs_df=lang_embs_df,
                               exclude_ids_df=exclude_ids_df[410:])                            
    print("Starting train loop.")
    train(model, 
          train_set,
          device,
          checkpoint_dir=checkpoint_dir,
          test_set=test_set,
          )