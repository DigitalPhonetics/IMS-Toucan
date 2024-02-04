import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import torch

def evaluate_all_models(csv_path, out_path, single_dim=False):
    df = pd.read_csv(csv_path, sep="|")
    np_dataset = df.to_numpy()
    if len(df.columns) > 50:
        X = np_dataset[:, :-16]
        y = np_dataset[:, -16:]
    else:
        X = np_dataset[:, :-1]
        y = np_dataset[:, -1:].ravel()

    lrregressor = LinearRegression()
    dtregressor = DecisionTreeRegressor()
    rdregressor = RandomForestRegressor()
    if single_dim:
        xgbregressor = XGBRegressor(objective='reg:squarederror')
    else:
        singletarget_multioutputregressor = MultiOutputRegressor(estimator=XGBRegressor(objective='reg:squarederror'))
        multitarget_multioutputregressor = XGBRegressor(objective="reg:squarederror",
                                                tree_method="hist", 
                                                multi_strategy="multi_output_tree")
    with open(out_path, "w") as f:
        f.write(f"csv_path: {csv_path}\n")
    lr_scores = cross_val_score(lrregressor, X, y, scoring="neg_mean_squared_error", cv=10, n_jobs=10)
    with open(out_path, "a") as f:
        f.write(f"LR scores: {lr_scores}\n")
        f.write(f"LR mean: {np.mean(lr_scores)}\n")
    dt_scores = cross_val_score(dtregressor, X, y, scoring="neg_mean_squared_error", cv=10, n_jobs=10)
    with open(out_path, "a") as f:
        f.write(f"DT scores: {dt_scores}\n")
        f.write(f"DT mean: {np.mean(dt_scores)}\n")
    rd_scores = cross_val_score(rdregressor, X, y, scoring="neg_mean_squared_error", cv=10, n_jobs=10)
    with open(out_path, "a") as f:
        f.write(f"RD scores: {rd_scores}\n")
        f.write(f"RD mean: {np.mean(rd_scores)}\n")
    if single_dim:
        xgb_scores = cross_val_score(xgbregressor, X, y, scoring="neg_mean_squared_error", cv=10, n_jobs=10)
        with open(out_path, "a") as f:
            f.write(f"XGBoost scores: {xgb_scores}\n")
            f.write(f"XGBoost mean: {np.mean(xgb_scores)}\n")        
    else:
        st_xgb_scores = cross_val_score(singletarget_multioutputregressor, X, y, scoring="neg_mean_squared_error", cv=10, n_jobs=10)
        with open(out_path, "a") as f:
            f.write(f"single-target XGBoost scores: {st_xgb_scores}\n")
            f.write(f"ST XGB mean: {np.mean(st_xgb_scores)}\n")
        mt_xgb_scores = cross_val_score(multitarget_multioutputregressor, X, y, scoring="neg_mean_squared_error", cv=10, n_jobs=10)
        with open(out_path, "a") as f:
            f.write(f"multi-target XGBoost scores: {mt_xgb_scores}\n")
            f.write(f"MT XGB mean: {np.mean(mt_xgb_scores)}\n")        


if __name__ == "__main__":
    # map, random, tree, asp
    for feature in ["map", "random", "tree", "asp"]:
        feature_csv_path = f"datasets/feature_dataset_{feature}_463_with_less_loss_fixed_tree_distance.csv"
        out_path = f"logs/multi_output_regression_{feature}.log"
        print(f"Evaluating {feature_csv_path}")
        evaluate_all_models(feature_csv_path, out_path, single_dim=False)

        # single-dim
        for i in range(16):
            feature_csv_path = f"datasets/single_dim/feature_dataset_{feature}_463_with_less_loss_fixed_tree_distance_dim{i}.csv"
            out_path = f"logs/single_dim/multi_output_regression_{feature}_dim{i}.log"
            print(f"Evaluating {feature_csv_path}")
            evaluate_all_models(feature_csv_path, out_path, single_dim=True)
    
    # COMBINED
    for cond in ["", "_individual_dists"]:
        feature_csv_path = f"datasets/feature_dataset_COMBINED_463_with_less_loss_fixed_tree_distance_average{cond}.csv"
        out_path = f"logs/multi_output_regression_COMBINED{cond}.log"
        print(f"Evaluating {feature_csv_path}")
        evaluate_all_models(feature_csv_path, out_path, single_dim=False)     

        # single-dim
        for i in range(16):
            feature_csv_path = f"datasets/single_dim/feature_dataset_COMBINED_463_with_less_loss_fixed_tree_distance_average{cond}_dim{i}.csv"
            out_path = f"logs/single_dim/multi_output_regression_COMBINED{cond}_dim{i}.log"
            print(f"Evaluating {feature_csv_path}")
            evaluate_all_models(feature_csv_path, out_path, single_dim=True)