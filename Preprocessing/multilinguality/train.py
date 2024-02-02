import torch
from sklearn.model_selection import KFold
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append("/home/behringe/hdd_behringe/IMS-Toucan")
from Preprocessing.multilinguality.lang_emb_dnn import LangEmbDataset, LangEmbPredictor, train
from Preprocessing.multilinguality.create_lang_emb_dataset import create_repeated_df
import datetime
import argparse


def added_noise_kfold_train_loop(csv_path, 
                                 checkpoint_dir, 
                                 log_dir, 
                                 n_repeats=100, 
                                 n_splits=10, 
                                 noise_std=0.01, 
                                 n_epochs: int = 10, 
                                 save_ckpt_every=10, 
                                 batch_size=4):
    """Train with k-fold cross-validation and save checkpoints to a specified dir.
    csv_path: str
    """
    os.makedirs(log_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_losses, val_losses = [], []
    log_path = os.path.join(log_dir, f"{os.path.basename(checkpoint_dir)}.log")
    with open(log_path, "w") as f:
        f.write(f"csv_path: {csv_path}\n")            
        f.write(f"train mode: noise_kfold | n_repeats: {n_repeats} | noise_std: {noise_std} | n_splits: {n_splits} | n_epochs: {n_epochs} | batch_size: {batch_size}\n")    
    df = pd.read_csv(csv_path, sep="|")
    repeated_df = create_repeated_df(df, n_repeats=n_repeats)

    # define manual k-fold splits (such that clusters are properly divided)
    n_larger_folds = len(repeated_df) % n_splits
    normal_shift = len(df) // n_splits * n_repeats
    larger_shift = normal_shift + n_repeats # add all samples of one more language
    test_start_idx = 0
    test_end_idx = 0
    # larger test folds
    for i in range(n_larger_folds):
        # get test fold
        test_end_idx += larger_shift
        test_df = repeated_df[test_start_idx:test_end_idx]
        # get train fold
        train_df_components = []
        if test_start_idx > 0:
            train_df_components.append(repeated_df[:test_start_idx])
        if test_end_idx < len(repeated_df):
            train_df_components.append(repeated_df[test_end_idx:])
        train_df = pd.concat(train_df_components)
        test_start_idx += larger_shift

        train_set = LangEmbDataset(dataset_df=train_df, add_noise=True, noise_std=noise_std)
        test_set = LangEmbDataset(test_df, add_noise=True, noise_std=noise_std)
        model = LangEmbPredictor(idim=17*5)
        print(f"Model {i+1}/{n_splits}")
        train_loss, val_loss = train(model, 
                                    train_set, 
                                    device, 
                                    checkpoint_dir,
                                    test_set, 
                                    batch_size=batch_size,
                                    n_epochs=n_epochs,
                                    save_ckpt_every=save_ckpt_every)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        with open(log_path, "a") as f:
            f.write(f"Model {i+1} | Train loss: {train_loss} | Val loss: {val_loss}\n")        

    # normal-length test folds
    for i in range(n_splits - n_larger_folds):
        # get test fold
        test_end_idx += normal_shift
        test_df = repeated_df[test_start_idx:test_end_idx]
        # get train fold
        train_df_components = []
        if test_start_idx > 0:
            train_df_components.append(repeated_df[:test_start_idx])
        if test_end_idx < len(repeated_df):
            train_df_components.append(repeated_df[test_end_idx:])
        train_df = pd.concat(train_df_components)
        test_start_idx += normal_shift

        train_set = LangEmbDataset(dataset_df=train_df, add_noise=True)
        test_set = LangEmbDataset(test_df, add_noise=True)
        model = LangEmbPredictor(idim=17*5)
        print(f"Model {i+1}/{n_splits}")
        train_loss, val_loss = train(model, 
                                    train_set, 
                                    device, 
                                    checkpoint_dir,
                                    test_set, 
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
    with open(log_path, "w") as f:
        f.write(f"csv_path: {csv_path}\n")
        f.write(f"train mode: kfold | n_splits: {n_splits} | n_epochs: {n_epochs} | batch_size: {batch_size}\n")    
    for i, (train_index, test_index) in enumerate(kfold.split(df)):
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]

        train_set = LangEmbDataset(dataset_df=train_df)
        test_set = LangEmbDataset(test_df)
        model = LangEmbPredictor(idim=17*5)
        print(f"Model {i+1}/{n_splits}")
        train_loss, val_loss = train(model, 
                                    train_set, 
                                    device, 
                                    checkpoint_dir,
                                    test_set=test_set, 
                                    batch_size=batch_size,
                                    n_epochs=n_epochs,
                                    save_ckpt_every=save_ckpt_every,
                                    model_id=i+1)
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
    with open(log_path, "w") as f:
        f.write(f"csv_path: {csv_path}\n")        
        f.write(f"train mode: full | n_epochs: {n_epochs} | batch_size: {batch_size}\n")    
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


def added_noise_full_train_loop():
    # TODO: to be implemented
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", help="path to dataset csv")
    parser.add_argument("--sep", default="|", help="delimiter for dataset csv")
    parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs to train")
    parser.add_argument("--n_splits", type=int, default=10, help="number of splits that should be created in kfold CV")
    parser.add_argument("--n_repeats", type=int, default=10, help="number of samples that should be generated (with added noise) for each original sample")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size for each model step")
    parser.add_argument("--checkpoint_dir", help="directory for saving checkpoints")
    parser.add_argument("--train_mode", choices=["kfold", "full", "noise_kfold", "noise_full"], default="kfold", help="choose training mode")
    parser.add_argument("--noise_std", type=float, default=0.01, help="standard deviation of the noise added to samples (if a `noise` train_mode is selected")
    args = parser.parse_args()
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else f"checkpoints/{timestamp}_{args.n_epochs}ep"
    log_dir = "logs"
    if args.train_mode == "full":
        print("Using all but 1 datapoints for training, only 1 test sample.")
        full_train_loop(csv_path=args.csv_path, checkpoint_dir=checkpoint_dir, log_dir=log_dir, n_epochs=args.n_epochs, batch_size=args.batch_size)
    elif args.train_mode == "kfold":
        print("Performing training with k-fold cross-validation.")
        kfold_train_loop(csv_path=args.csv_path, 
                         checkpoint_dir=checkpoint_dir, 
                         log_dir=log_dir, 
                         n_epochs=args.n_epochs, 
                         n_splits=args.n_splits, 
                         batch_size=args.batch_size)
    elif args.train_mode == "noise_kfold":
        print("Performing training with increased, noise-augmented dataset and k-fold cross-validation.")
        added_noise_kfold_train_loop(csv_path=args.csv_path, 
                                     checkpoint_dir=checkpoint_dir, 
                                     log_dir=log_dir, 
                                     n_epochs=args.n_epochs, 
                                     n_splits=args.n_splits, 
                                     batch_size=args.batch_size,
                                     noise_std=args.noise_std,
                                     n_repeats=args.n_repeats)

