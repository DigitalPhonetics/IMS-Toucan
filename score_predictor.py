import kan
import torch
import random
import wandb
import json
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import KFold
import time
import math
import matplotlib.pyplot as plt
import seaborn as sns
import sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')  # Redirect stderr to suppress tqdm

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

class Scorer():

    def __init__(self, config, path):
        if config is None:
            self.load_model(path)
        else:
            self.config = copy.deepcopy(config)
            self.predictor = kan.KAN(width=config['width'], grid=config['grid'], k=config['k'], seed=config['seed'])

    def save_model(self, path, config_in_name=True):
        if config_in_name:
            name = path + '_'.join([str(s) for s in self.config['width']]) + "-g" + str(self.config['grid']) + "-k" + str(self.config['k']) + "-" + '-'.join([dim.replace("_zscore", "") for dim in self.config['inputs']]) + '.pt'
        else:
            name = path + '.pt'
        torch.save({ "model"       : self.predictor.state_dict(),
                        "config"      : self.config
                    }, name)
        
    def load_model(self, path):
        state = torch.load(path) # , map_location=torch.device('cpu'))
        # remove 0s [[5,0], [3,0], [1,0]] -> [5,3,1]
        #state['config']['width'] = [w[0] for w in state['config']['width']]
        config = state['config']
        self.config = copy.deepcopy(config) # otherwise the config will be changed in the model
        model = kan.KAN(width=config['width'], grid=state['config']['grid'], k=state['config']['k'], seed=state['config']['seed'])
        model.load_state_dict(state['model'])
        self.predictor = model

    def return_model_from_path(self, path):
        state = torch.load(path, map_location=torch.device('cpu'))
        # remove 0s [[5,0], [3,0], [1,0]] -> [5,3,1]
        #state['config']['width'] = [w[0] for w in state['config']['width']]
        config = state['config']
        model = kan.KAN(width=config['width'], grid=state['config']['grid'], k=state['config']['k'], seed=state['config']['seed'])
        model.load_state_dict(state['model'])
        return model

    def generate_dummy_data(self, x, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        
        # Step 1: Generate random output values between 1 and 5 for each sample
        outputs = torch.empty(x, 1).uniform_(1, 5)
        
        # Step 2: Create input array of shape [x, 5] with values around the mean of the outputs
        means = outputs.expand(-1, 5)
        inputs = torch.normal(mean=means, std=0.5)

        # Step 3: Scale the input values to be between 0 and 1
        inputs_min = inputs.min()
        inputs_max = inputs.max()
        scaled_inputs = (inputs - inputs_min) / (inputs_max - inputs_min) * 0.9 + 0.05
        
        return scaled_inputs, outputs

    def load_data(self, wandb_id, validation=False, with_variances=False):
        
        if not os.path.exists(f'samples/{wandb_id}/Evaluation_Table.table.json'):
            run = wandb.init(id=wandb_id, project="IMS-Toucan-Prosody-Variance", entity="prosody-variance", resume='allow')
            # Fetch the logged table
            artifact = run.use_artifact(f'prosody-variance/IMS-Toucan-Prosody-Variance/run-{wandb_id}-Evaluation_Table:v1', type='run_table')
            artifact.download(root=f"samples/{wandb_id}")
            run.finish()
            
        
        with open(f'samples/{wandb_id}/Evaluation_Table.table.json', 'r') as f:
            table_data = json.load(f)

        # Convert W&B Table to a pandas dataframe
        columns = table_data['columns']
        data = table_data['data']
        df = pd.DataFrame(data=data, columns=columns)
        if with_variances:
            labels = df[['self_test', 'self_test_variance']].values

            labels = torch.tensor(labels.astype(np.float64)) # convert to tensor
        else:
            labels = df['self_test'].values
            #labels[labels == '-'] = '0' # replace missing values
            labels = torch.tensor(labels.astype(np.float64)).unsqueeze(1) # convert to tensor

        inputs = df[self.config['inputs']].values
        inputs = torch.tensor(inputs.astype(np.float64)) # convert to tensor

        # Generate indices for the split
        indices = torch.randperm(labels.size(0))

        if validation:
            # split into 3 non overlaping parts
            split_idx = int(0.6 * labels.size(0))
            val_idx = int(0.8 * labels.size(0))

            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:val_idx]
            test_indices = indices[val_idx:]
            
            train_labels = labels[train_indices]
            val_labels = labels[val_indices]
            test_labels = labels[test_indices]
            train_inputs = inputs[train_indices]
            val_inputs = inputs[val_indices]
            test_inputs = inputs[test_indices]

            print(f"Train: {train_inputs.size(0)}, Val: {val_inputs.size(0)}, Test: {test_inputs.size(0)}")
            dataset = {
            'train_input': train_inputs,
            'val_input': val_inputs,
            'test_input' : test_inputs,
            'train_label': train_labels,
            'val_label': val_labels,
            'test_label': test_labels
            }
        
        else:
        # Calculate the split index
            split_idx = int(0.8 * labels.size(0))

            # Split indices into training and validation
            train_indices = indices[:split_idx]
            test_indices = indices[split_idx:]

            train_labels = labels[train_indices]
            test_labels = labels[test_indices]
            train_inputs = inputs[train_indices]
            test_inputs = inputs[test_indices]

            dataset = {
            'train_input': train_inputs,
            'test_input' : test_inputs,
            'train_label': train_labels,
            'test_label': test_labels
            }
        return dataset

    def  cross_val_score(self, model_path, dataset, k=5):
        # split data into k parts
        kf = KFold(n_splits=k)
        # set last loss to infinity
        last_loss = 1000000
        metrics = []
        for train_index, test_index in kf.split(dataset['train_input']):
            train_input, test_input = dataset['train_input'][train_index], dataset['train_input'][test_index]
            train_label, test_label = dataset['train_label'][train_index], dataset['train_label'][test_index]
            k_dataset = {
                'train_input': train_input,
                'test_input' : test_input,
                'train_label': train_label,
                'test_label': test_label
            }
            # Train new model
            current_model = self.return_model_from_path(model_path) 
            for i in range(100):
                current_model.fit(k_dataset, steps=1)
                # eval model
                predicted = self.predictor(test_input)
                loss = torch.nn.functional.mse_loss(predicted, test_label)
                print(f"lasltloss{last_loss} Loss: {loss.item()}")
                # early stopping
                if last_loss <= loss.item():
                    break
                last_loss = loss.item()
            metrics.append(last_loss)
        return sum(metrics)/len(metrics)

    def train(self, path, dataset, steps=100, k=20):
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(path + "Models/"):
            os.makedirs(path + "Models/")
        
        # save model before training
        self.save_model(path + "Models/-1", config_in_name=False)

        # check if there is a validation set
        if 'val_input' in dataset:
            last_loss = torch.nn.functional.mse_loss(self.predictor(dataset['val_input']), dataset['val_label'])
        else:
            with HiddenPrints():
                last_loss = self.cross_val_score(path + "Models/-1.pt", dataset, k=k)
            # cross validation+

        # tqdm bar for training
        
        with tqdm(total=steps, desc="Evaluating Models") as pbar:
            for i in range(steps):
                with HiddenPrints():
                    self.predictor.fit(dataset, steps=1)
                # eval model
                if 'val_input' in dataset:
                    predicted = self.predictor(dataset['val_input'])
                    loss = torch.nn.functional.mse_loss(predicted, dataset['val_label'])
                    loss = loss.item()
                else:
                    with HiddenPrints():
                        loss = self.cross_val_score(f"{path}Models/{i -1}.pt", dataset, k=k)
                pbar.set_description(f"Step: {i}, Loss: {loss}, Last Loss: {last_loss}")
                pbar.refresh()
                # early stopping
                
                if last_loss <= loss - 0.0001:
                    self.save_model(path)
                    return
                # save model
                self.save_model(path + f"Models/{i}", config_in_name=False) 

                last_loss = loss
                pbar.update(1)
        self.save_model(path)

    def predict(self, df):
        inputs = df[self.config['inputs']].values
        inputs = torch.tensor(inputs.astype(np.float64))

        score = self.predictor(inputs)

        return score

def eval(path, version, dataset_wandb, use_wandb=True, with_variances=False , validation=False):
    # initialize wandb

    if use_wandb:
        wandb.init(project="IMS-Toucan-Prosody-Variance", entity="prosody-variance",
                   name=f"KAN_Evaluation_{version}_{time.strftime('%Y%m%d-%H%M%S')}")
    if not os.path.exists(f"visualizations/{version}/Evaluation_Metrics.csv"):
        data = {
        'Model': ["-".join(model.split("-")[:3]) for model in os.listdir(path)],
        'mses': [],
        'rmses': [],
        'group': []
        }
        # remove "Model" entry from "Model" column
        data['Model'].remove("Models")
        print(len(data['Model']))
        # go through all models in the Models folder
        with tqdm(total=len(os.listdir(path)) - 1, desc="Evaluating Models") as outer_bar:
            for model in os.listdir(path):
                if model == "Models":
                    continue
                outer_bar.set_description(f"Evaluating model: {model[:-3]}")
                outer_bar.refresh()
                # load model
                scorer = Scorer(None, path + model)

                # load data
                dataset = scorer.load_data(dataset_wandb, validation=validation, with_variances=with_variances)
                print(dataset['test_label'])

                # eval model
                predicted = scorer.predictor(dataset['test_input'])

                # calculate metrics
                MSE = torch.nn.functional.mse_loss(predicted, dataset['test_label'])
                RMSE = torch.sqrt(MSE).item()
                MSE = MSE.item()

                data['mses'].append(MSE)
                data['rmses'].append(RMSE)
                
                if 'harmonic' in model:
                    if 'mean' in model:
                        data['group'].append('harmonic-both')
                    else:
                        data['group'].append('harmonic-var')
                elif 'geometric' in model:
                    if 'mean' in model:
                        data['group'].append('geometric-both')
                    else:
                        data['group'].append('geometric-var')
                elif 'arithmetic' in model:
                    if 'mean' in model:
                        data['group'].append('arithmetic-both')
                    else:
                        data['group'].append('arithmetic-var')
                elif 'sum' in model:
                    if 'mean' in model:
                        data['group'].append('sum-both')
                    else:
                        data['group'].append('sum-var')
                else:
                    if 'mean' in model:
                        data['group'].append('separate-both')
                    else:
                        data['group'].append('separate-var')

                # create folder for visualizations
                if not os.path.exists(f"visualizations/{version}"):
                    os.makedirs(f"visualizations/{version}")
                # plot results
                if scorer.config['width'][-1]  == 1:
                    scorer.predictor.plot(path_to_save=f"visualizations/{version}/{model[:-3]}.png", in_vars=[i[:-7] for i in scorer.config['inputs']], out_vars=['Score'], varscale=0.3, scale=1, title=f"test{model[:-3]}", beta=2)
                else:
                    scorer.predictor.plot(path_to_save=f"visualizations/{version}/{model[:-3]}.png", in_vars=scorer.config['inputs'], out_vars=['Score', 'Variance'], varscale=1, scale=1, title=f"{model[:-3]}")
                plt.close()
                if use_wandb:
                    # save results to wandb 
                    wandb.log({f"Visual Representation of {model[:-3]}": wandb.Image(f"visualizations/{version}/{model[:-3]}.png")})
                
                outer_bar.update(1)
        # plot results with plt
        # save result as table
        results = pd.DataFrame(data)
        print(results)

        results = results.sort_values(by='mses', ascending=False).copy(deep=True)
        # order after 1 in name 5-7-1 -> 5, then get index accordingly
    else:
        results = pd.read_csv(f"visualizations/{version}/Evaluation_Metrics.csv")
    """
    def custom_sort_key(hue_value):
        # Split the string into integers and return as a tuple for sorting
        print(hue_value)
        print(hue_value.split("-")[0])
        return tuple(map(int, hue_value.split("-")[0].split('_')))
    # Get unique hue values and sort them using the custom key
    sorted_hues = sorted(results['Model'].unique(), key=custom_sort_key, reverse=True)
    print(results)
    """
    # two subplots
    n_models = len(results['group'].unique())
    cols = math.ceil(math.sqrt(n_models))  # Number of columns
    rows = math.ceil(n_models / cols)      # Number of rows

    # add hue_value eg. 5-3-1_name-name2... -> 3-1, 5-1_name-name2... -> 1
    # first use name.split("_")[0] to get all the numbers, then split by "-" and take the all numbers but first
    results['hue'] = results['Model'].apply(lambda x: "-".join(x.split("-")[0].split("_")[1:]))
    hue_order = sorted(results['hue'].unique())
                                                  
    # 2. Create the grid of subplots
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), sharey=False)
    # set title
    fig.suptitle('MSE - lower is better', fontsize=26)
    axes = axes.flatten()  # Convert axes to 1D array
    for i, group in enumerate(results['group'].unique()):
        current_results = results[results['group'] == group]
        sns.barplot(x='Model', y='mses', hue='hue', data=current_results, hue_order=hue_order, ax=axes[i])
        axes[i].set_title(group)
        axes[i].set_xlabel("Model Size")
        axes[i].set_ylabel("MSE")
        plt.setp(axes[i].get_xticklabels(), rotation=45, ha="right")
    
    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(f"visualizations/{version}/Evaluation_Metrics.png")
    plt.close()
    if use_wandb:
        wandb.log({"Evaluation Metrics": wandb.Image(f"visualizations/{version}/Evaluation_Metrics.png")})
    
    
    # 1. Find the indices of the rows with the minimum MSE for each group
    best_indices = results.groupby('group')['mses'].idxmin()

    # 2. Extract the 'Group' and corresponding 'Models' using these indices
    best_group_model = results.loc[best_indices, ['group', 'Model']].reset_index(drop=True)

    # 3. Rename the 'Models' column to 'best_name'
    best_group_model = best_group_model.rename(columns={'Model': 'best_name'})

    # The resulting DataFrame
    print(best_group_model)
    best_mse = results.groupby('group')['mses'].min().reset_index()
    result_df = pd.merge(best_mse, best_group_model, left_on='group', right_on='group') 
    print(result_df)
    # make hue for first int in name

    plt.figure(figsize=(7, 5))
    sns.barplot(data=result_df, x='group', y='mses', hue='best_name')
    plt.title('Best MSE for Each Model')
    plt.xlabel('Models')
    plt.ylabel('MSE')
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(f"visualizations/{version}/Best_MSE.png")
    plt.close()
    if use_wandb:
        wandb.log({"Best MSE": wandb.Image(f"visualizations/{version}/Best_MSE.png")})

    results.to_csv(f"visualizations/{version}/Evaluation_Metrics.csv", index=False)
    table = wandb.Table(data=results)
    # save results as wandb table
    if use_wandb:
        wandb.log({"KAN Eval": table})

def train_all(path, wandb_id, with_variances=False , cross_val=False):
    pd.set_option('display.max_columns', None)

    inputs = [['distance_speaker_zscore', 'gmm_zscore', 'wv_mos_zscore', 'overlap_zscore', 'var_sum_score'],
              ['distance_speaker_zscore', 'gmm_zscore', 'wv_mos_zscore', 'overlap_zscore', 'var_arithmetic_score'],
                ['distance_speaker_zscore', 'gmm_zscore', 'wv_mos_zscore', 'overlap_zscore', 'var_geometric_score'],
                ['distance_speaker_zscore', 'gmm_zscore', 'wv_mos_zscore', 'overlap_zscore', 'var_harmonic_score'],
                ['distance_speaker_zscore', 'gmm_zscore', 'wv_mos_zscore', 'overlap_zscore', 'var_sum_score', 'mean_sum_score'],
                ['distance_speaker_zscore', 'gmm_zscore', 'wv_mos_zscore', 'overlap_zscore', 'var_arithmetic_score', 'mean_arithmetic_score'],
                ['distance_speaker_zscore', 'gmm_zscore', 'wv_mos_zscore', 'overlap_zscore', 'var_geometric_score', 'mean_geometric_score'],
                ['distance_speaker_zscore', 'gmm_zscore', 'wv_mos_zscore', 'overlap_zscore', 'var_harmonic_score', 'mean_harmonic_score'],
                ['distance_speaker_zscore', 'gmm_zscore', 'wv_mos_zscore', 'overlap_zscore', 'pitch_var', 'energy_var', 'duration_var'],
                ['distance_speaker_zscore', 'gmm_zscore', 'wv_mos_zscore', 'overlap_zscore', 'pitch_var', 'energy_var', 'duration_var', 'pitch_mean', 'energy_mean', 'duration_mean']]

            

    for input in inputs:
    # Load data [[5,1,1], [5,3,1], [5,7,1], [5,1]]:
        if with_variances:
            widths = [[len(input),1,2], [len(input),3,2], [len(input),7,2], [len(input),2]]
        else:
            widths = [[len(input),1,1], [len(input),3,1], [len(input),7,1], [len(input),1]]
        for width in widths:
            config = {
                'width': width,
                'grid': 3,
                'k': 3,
                'seed': 42,
                'inputs': input
            }

            print(config)
            scorer = Scorer(config, None)
            dataset = scorer.load_data(wandb_id, validation=(not cross_val), with_variances=with_variances)
            
            # Train model
            scorer.train(path, dataset, steps=100)
    

if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    
    train_all("Models/CV_Var/", "wwzmrcye" ,with_variances=True, cross_val=True)
    eval("Models/CV_Var/", "KAN-CV_Var", "wwzmrcye", use_wandb=False, with_variances=True, validation=True)

 

