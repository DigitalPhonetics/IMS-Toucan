import torch
from sklearn.model_selection import KFold
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append("/home/behringe/hdd_behringe/IMS-Toucan")
from Preprocessing.multilinguality.lang_emb_dnn import LangEmbDataset, LangEmbPredictor, train
import datetime
import argparse



def kfold_train_loop(csv_path, checkpoint_dir, log_dir, n_splits=10, n_epochs: int = 10, save_ckpt_every=10, batch_size=4):
    """Train with k-fold cross-validation and save checkpoints to a specified dir.
    csv_path: str
    """
    os.makedirs(log_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(csv_path, sep="|")
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    train_losses, val_losses = [], []
    log_path = os.path.join(log_dir, f"{os.path.basename(checkpoint_dir)}.log")
    for i, (train_index, test_index) in enumerate(kfold.split(df)):
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]

        train_set = LangEmbDataset(dataset_df=train_df)
        test_set = LangEmbDataset(test_df)
        model = LangEmbPredictor(idim=17*5)
        print(f"Model {i+1}/{n_splits}")
        train_loss, val_loss = train(model, 
                                    train_set, 
                                    test_set, 
                                    device, 
                                    checkpoint_dir,
                                    batch_size=batch_size,
                                    n_epochs=n_epochs,
                                    save_ckpt_every=save_ckpt_every)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        with open(log_path, "a") as f:
            f.write(f"Model {i+1} | Train loss: {train_loss} | Val loss: {val_loss}\n")
    avg_train_loss = sum(train_losses)/len(train_losses)
    avg_val_loss = sum(val_losses)/len(val_losses)
    with open(log_path, "a") as f:
        f.write(f"Summary | Average train loss: {avg_train_loss} | Average val loss: {avg_val_loss}\n")
    return avg_train_loss, avg_val_loss

def full_train_loop(csv_path, checkpoint_dir, log_dir, n_epochs: int = 10, save_ckpt_every=10, batch_size=4):
    """Train on all but 1 datapoints (with only 1 test sample) and save checkpoints to a specified dir.
    csv_path: str
    """
    os.makedirs(log_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_df = pd.read_csv(csv_path, sep="|")[:-1]

    train_losses = []
    log_path = os.path.join(log_dir, f"{os.path.basename(checkpoint_dir)}.log")

    train_set = LangEmbDataset(dataset_df=train_df)

    model = LangEmbPredictor(idim=17*5)
    train_loss = train(model, 
                        train_set, 
                        device, 
                        checkpoint_dir,
                        batch_size=batch_size,
                        n_epochs=n_epochs,
                        save_ckpt_every=save_ckpt_every)
    train_losses.append(train_loss)
    with open(log_path, "a") as f:
        f.write(f"Train loss: {train_loss}\n")
    avg_train_loss = sum(train_losses)/len(train_losses)
    with open(log_path, "a") as f:
        f.write(f"Summary | Average train loss: {avg_train_loss}\n")
    return avg_train_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", help="path to dataset csv")
    parser.add_argument("--sep", default="|", help="delimiter for dataset csv")
    parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs to train")
    parser.add_argument("--checkpoint_dir", help="directory for saving checkpoints")
    parser.add_argument("--full_train", action="store_true", help="if set to True, will use all but 1 datapoints for training (single test sample)")
    args = parser.parse_args()
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else f"checkpoints/{timestamp}_{args.n_epochs}ep"
    log_dir = "logs"
    if args.full_train:
        print("Using all but 1 datapoints for training, only 1 test sample.")
        full_train_loop(csv_path=args.csv_path, checkpoint_dir=checkpoint_dir, log_dir=log_dir, n_epochs=args.n_epochs)
    else:
        print("Performing training with k-fold cross-validation.")
        kfold_train_loop(csv_path=args.csv_path, checkpoint_dir=checkpoint_dir, log_dir=log_dir, n_epochs=args.n_epochs)

