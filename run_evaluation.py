import re
import time
import wandb
import argparse
import os
import json
import pandas as pd
import test_suite as test
import torch
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np
from tqdm import tqdm
from scipy.stats import hmean, gmean
from score_predictor import Scorer

class HiddenPrints:
    # hides prints from the called functions
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def run_eval(path_to_models, version, use_wandb=True, gpu_id=None, per_sample=False, multi_speaker=False):
    """
    Runs the evaluation

    path_to_models (String): points to saved models that will be included in the eval
    version (String): name that is associated with this run
    use_wandb (Boolean): upload evaluation results to wandb
    gpu_id (Integer): specifies which gpu is used
    per_sample (Boolean): specifies wether the values are summarized per sample or acroos TODO: only FALSE is used
    multi_speaker (Boolean): use of multiply speakers that the models are compared to
    
    """
    torch.set_deterministic_debug_mode(False)
    # Suppress specific warning from transformers configuration
    warnings.filterwarnings("ignore", message="Passing `gradient_checkpointing` to a config initialization is deprecated")
    warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

    if gpu_id is None:
        device = torch.device("cpu")
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
        device = torch.device("cuda")

    if use_wandb:
        wandb.init(name=f"Evaluation_{time.strftime('%Y%m%d-%H%M%S')}")

    if per_sample:
        results = {
            'model': [],
            'self_test': [],
            'distance_speaker': [],
            'overlap': [],
            'gmm': [],
            'pitch_mean': [],
            'pitch_var': [],
            'energy_mean': [],
            'energy_var': [],
            'duration_mean': [],
            'duration_var': [],
            'wv_mos': [],
            'predicted': []
            }
    else:
        results = {
            'model': [],
            'self_test': [],
            'self_test_variance': [],
            'distance_speaker': [],
            'overlap': [],
            'gmm': [],
            'mean_sum_score': [],
            'var_sum_score': [],
            'mean_arithmetic_score': [],
            'var_arithmetic_score': [],
            'mean_geometric_score': [],
            'var_geometric_score': [],
            'mean_harmonic_score': [],
            'var_harmonic_score': [],
            'pitch_mean': [],
            'pitch_var': [],
            'energy_mean': [],
            'energy_var': [],
            'duration_mean': [],
            'duration_var': [],
            'wv_mos': [],
            'predicted': []
            }
    
    model_ids = os.listdir(path_to_models)
    print("Eval models:")
    print(model_ids)

    total_length = len(model_ids)
    counter = 0
    if not os.path.exists(f"samples/evaluation_{version}.csv"):
        with tqdm(total=total_length, desc="Evaluating Models") as outer_bar:
            for model_id in model_ids:
                temperatures = [0.4,0.8]#[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
                if "NF" in model_id:
                    architecture = "NF"
                elif "CFM" in model_id:
                    architecture = "CFM"
                elif "RF" in model_id:
                    architecture = "RF"
                elif "DET" in model_id:
                    architecture = "DET"
                else:
                    architecture = "CFM"
                with tqdm(total=len(temperatures), desc="Start") as inner_bar:
                    for temp in temperatures:

                        full_name = model_id + "_" + str(temp)
                        outer_bar.set_description(f"Evaluating Model {full_name}")
                        outer_bar.refresh()
                        full_model_path = path_to_models + "/" + model_id + "/best.pt"
                        audio_samples=16
                        
                        if not os.path.exists(f"audios/{version}/{full_name}/{full_name}-0.wav"):
                            # create sample audios
                            inner_bar.set_description("Creating sample audios")
                            inner_bar.refresh()
                            with HiddenPrints():
                                test.variance_test(f"{full_name}", dir=str(version), samples= audio_samples, model_id=full_model_path, exec_device=device, speaker_reference="audios/RAVDESS_one/Actor_19/03-01-01-01-01-01-19.wav", prosody_creativity=temp, architecture=architecture)
                        # TODO: variance test good name?

                        # compute distance to speaker and overlap
                        inner_bar.set_description("Compare to speaker")
                        inner_bar.refresh()
                        if multi_speaker:
                            reference_speaker = [f"audios/RAVDESS_all/{file}" for file in os.listdir("audios/RAVDESS_all") if os.path.isfile(os.path.join("audios/RAVDESS_all", file))]
                            print(reference_speaker)
                        else:
                            reference_speaker="audios/RAVDESS_one/Actor_19/03-01-01-01-01-01-19.wav"
                        with HiddenPrints():
                            speaker_distance, speaker_overlap = test.compare_to_reference(version=f"{full_name}", 
                                                                                        reference_speaker=reference_speaker,
                                                                                        samples=16 ,model_id=full_model_path, device=device, use_wandb=use_wandb,
                                                                                        prosody_creativity=temp, number = counter,
                                                                                            architecture=architecture, eval_name=version)
                        counter += 1 # set counter for slider in wandb

                        # compute statistics
                        inner_bar.set_description("Computing Variance score")
                        inner_bar.refresh()
                        with HiddenPrints():
                            pitch_mean, pitch_std, energy_mean, energy_std, duration_mean, duration_std = test.get_mean_var(f"samples/{full_name}_data_samples_sentence.csv", per_sample=per_sample)
                        

                        if not per_sample:
                            mean_sum_score = pitch_mean.values[0] + energy_mean.values[0] + duration_mean.values[0] 
                            variance_sum_score = pitch_std.values[0] +  energy_std.values[0] + duration_std.values[0]

                            mean_ar_score = mean_sum_score / 3 
                            variance_ar_score = variance_sum_score / 3

                            mean_list = [pitch_mean.values[0], energy_mean.values[0], duration_mean.values[0]]
                            variance_list = [pitch_std.values[0], energy_std.values[0], duration_std.values[0]]
                            mean_geo_score = gmean(mean_list)
                            variance_geo_score = gmean(variance_list)
                            
                            mean_har_score = hmean(mean_list)
                            variance_har_score = hmean(variance_list)


                        # compute GMM score
                        inner_bar.set_description("Computing GMM score")
                        inner_bar.refresh()
                        with HiddenPrints():
                            gmm_score = test.compare_speaker_gmm(path_to_model_data=f"samples/{full_name}_data_samples_sentence.csv", path_to_speaker_data="samples/speaker.csv")    
                        
                        # compute automatic mos score
                        inner_bar.set_description("Computing MOS score")
                        inner_bar.refresh()
                        #dir_names = set([dir_name for dir_name in os.listdir(f"audios/{version}/") if model_id == dir_name[:-4]])
                        with HiddenPrints():
                            mos_score = test.get_automatic_mos_score(f"audios/{version}/{full_name}", device = device, per_sample=per_sample)

                        inner_bar.set_description(f"Finished model {full_name}")
                        inner_bar.refresh()
                        
                        if per_sample:
                            i = 0
                            for mos, p_m, p_v, e_m, e_v, d_m, d_v in zip(mos_score, pitch_mean, pitch_std, energy_mean, energy_std, duration_mean, duration_std):
                               
                                results["model"].append(full_name + "-" + str(i))
                                i += 1
                                results["self_test"].append(-1)
                                results["distance_speaker"].append(speaker_distance)
                                results["overlap"].append(speaker_overlap)
                                results["gmm"].append(gmm_score)
                                results["pitch_mean"].append(p_m)
                                results["pitch_var"].append(p_v)
                                results["energy_mean"].append(e_m)
                                results["energy_var"].append(e_v)
                                results["duration_mean"].append(d_m)
                                results["duration_var"].append(d_v)
                                results["wv_mos"].append(mos)
                                results["predicted"].append(-1) # kan predicts later

                        else:
                            outer_bar.write(f"Summary model {full_name}:")
                            outer_bar.write(f"gmm_score : {gmm_score}")  
                            outer_bar.write(f"Speaker Score:{speaker_distance}")
                            outer_bar.write(f"Speaker overlap: {speaker_overlap}")
                            outer_bar.write(f"mean sum score: {mean_sum_score}")
                            outer_bar.write(f"var sum score: {variance_sum_score}")
                            outer_bar.write(f"mean arithmetic score: {mean_ar_score}")
                            outer_bar.write(f"var arithmetic score: {variance_ar_score}")
                            outer_bar.write(f"mean geometric score: {mean_geo_score}")
                            outer_bar.write(f"var geometric score: {variance_geo_score}")
                            outer_bar.write(f"mean harmonic score: {mean_har_score}")
                            outer_bar.write(f"var harmonic score: {variance_har_score}")
                            outer_bar.write(f"pitch_mean: {pitch_mean.values[0]}")
                            outer_bar.write(f"pitch_var: {pitch_std.values[0]}")
                            outer_bar.write(f"energy_mean: {energy_mean.values[0]}")
                            outer_bar.write(f"energy_var: {energy_std.values[0]}")
                            outer_bar.write(f"duration_mean: {duration_mean.values[0]}")
                            outer_bar.write(f"duration_var: {duration_std.values[0]}")
                            outer_bar.write(f"Mos: {mos_score}")
                            outer_bar.write("#"*30)

                            results["model"].append(full_name)
                            results["self_test"].append(-1)
                            results["self_test_variance"].append(-1)
                            results["distance_speaker"].append(speaker_distance)
                            results["overlap"].append(speaker_overlap)
                            results["gmm"].append(gmm_score)
                            results["mean_sum_score"].append(mean_sum_score)
                            results["var_sum_score"].append(variance_sum_score)
                            results["mean_arithmetic_score"].append(mean_ar_score)
                            results["var_arithmetic_score"].append(variance_ar_score)
                            results["mean_geometric_score"].append(mean_geo_score)
                            results["var_geometric_score"].append(variance_geo_score)
                            results["mean_harmonic_score"].append(mean_har_score)
                            results["var_harmonic_score"].append(variance_har_score)
                            results["pitch_mean"].append(pitch_mean.values[0])
                            results["pitch_var"].append(pitch_std.values[0])
                            results["energy_mean"].append(energy_mean.values[0])
                            results["energy_var"].append(energy_std.values[0])
                            results["duration_mean"].append(duration_mean.values[0])
                            results["duration_var"].append(duration_std.values[0])
                            results["wv_mos"].append(mos_score)
                            results["predicted"].append(-1) # kan predicts later


                        if use_wandb:
                            wandb.log({
                                'Outer': outer_bar.n,
                                'Inner': inner_bar.n
                            })

                        inner_bar.update(1)
                outer_bar.update(1)
                    
        # Create DataFrame for visualizations and normalization
        df = pd.DataFrame(results)
        pd.set_option('display.max_columns', 18)

        # Metrics
        if per_sample:
            metrics = ['distance_speaker', 'gmm', 'wv_mos', 'overlap', 'pitch_mean', 'pitch_var', 'energy_mean',
                    'energy_var', 'duration_mean', 'duration_var']
        else:
            metrics = ['distance_speaker', 'gmm', 'wv_mos', 'overlap', 'mean_sum_score', 'var_sum_score',
                    'mean_arithmetic_score', 'var_arithmetic_score', 'mean_geometric_score', 'var_geometric_score',
                    'mean_harmonic_score', 'var_harmonic_score', 'pitch_mean', 'pitch_var', 'energy_mean',
                    'energy_var', 'duration_mean', 'duration_var']
        
        # Standardize the metrics
        for metric in metrics:
            if metric == "distance_speaker":  # Bhattacharyya distance (lower is better)
                df[metric + "_transformed"] = -df[metric].abs()  # Negate to treat lower as better
            elif metric == "gmm":  # Lower is better
                df[metric + "_transformed"] = -df[metric]  # Negate to treat lower as better
            elif metric == "overlap":  # Should be close to 1
                df[metric + "_transformed"] = -(df[metric] - 1).abs()  # Negate the absolute deviation from 1
            else:  # Higher is better (variance and wv_mos)
                df[metric + "_transformed"] = df[metric]
        
        
        # Normalize each metric to z-scores
        for metric in metrics:
            transformed_metric = metric + "_transformed"
            df[metric + "_zscore"] = (df[transformed_metric] - df[transformed_metric].mean()) / df[transformed_metric].std()
        """
        for metric in metrics:
            transformed_metric = metric + "_transformed"
            min_value = df[transformed_metric].min()
            max_value = df[transformed_metric].max()
            
            # Min-Max scaling to range [0, 1]
            # name should be changed
            df[metric + "_zscore"] = (df[transformed_metric] - min_value) / (max_value - min_value)
        """
        df.to_csv(f"samples/evaluation_{version}.csv")
    else: 
        # load evaluation data
        df = pd.read_csv(f"samples/evaluation_{version}.csv")

    # if kan is trained total score can be predicted
    if per_sample:
        path_to_predictor = "Models/KAN/sample_10_1_1-g3-k3-distance_speaker-gmm-wv_mos-overlap-pitch_var-energy_var-duration_var-pitch_mean-energy_mean-duration_mean.pt"
    else:
        #path_to_predictor = "Models/KAN/5_7_1-g3-k3-distance_speaker-gmm-wv_mos-overlap-var_arithmetic_score.pt"
        path_to_predictor = "Models/KAN/6_7_2-g3-k3-distance_speaker-gmm-wv_mos-overlap-var_geometric_score-mean_geometric_score.pt"
    
    df['predictor'] = [path_to_predictor for _ in range(len(df['predicted']))]
    
    default_type = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    scorer = Scorer(config=None, path=path_to_predictor)
    score = scorer.predict(df)
    pd.set_option('display.max_columns', 60)
    print(df)
    with_variance = True if score[0].shape[-1] == 2 else False

    torch.set_default_dtype(default_type)
    print(with_variance)
    if with_variance:
        df["predicted"] = score[:,0].detach().numpy()
        df["predicted_var"] = score[:,1].detach().numpy()
    else:
        df["predicted"] = score.detach().numpy()

    # Sort the DataFrame by the total score
    df_sorted = df.sort_values(by='predicted', ascending=False).copy(deep=True)
    if use_wandb:
        table = wandb.Table(dataframe=df_sorted)
        wandb.log({"Evaluation_Table": table})
    
    # save used data as csv
    path_to_predictor = path_to_predictor.replace("/", "-")
    if per_sample:
        df_sorted['model'] = df_sorted['model'].apply(lambda x: '_'.join(x.split('_')[:-1]))


    # aggregate data
    #aggregations = ['all', 'model', 'temp', 'drop', 'log', 'order']
    #aggregations = ['channels', 'layers', 'kernal', 'temp', 'all']
    aggregations = ['architecture', 'all']

    # Select the required columns
    df_metrics = df_sorted[['model', 'wv_mos', 'distance_speaker', 'overlap', 'gmm', "predicted", 'pitch_mean', 'duration_mean']]

    # Save to CSV (not grouped, keeping original values)
    csv_path = f"visualizations/{version}_metrics.csv"
    df_metrics.to_csv(csv_path, index=False)

    print(f"CSV saved at: {csv_path}")
    
    df_sorted['group'] = df_sorted['model'].apply(lambda x: x.split('_')[0])
    model_order = {"NF": 1, "CFM": 2, "RF": 3}
    df_sorted['sort_key'] = df_sorted['group'].map(model_order)

    # Extract numerical part from model names
    def extract_numeric_part(model_name):
        match = re.search(r'[-+]?\d*\.\d+|\d+', model_name)  # Finds a floating-point or integer number
        return float(match.group()) if match else float('inf')  # Assigns 'inf' if no number is found (to keep them last)

    df_sorted['numeric_part'] = df_sorted['model'].apply(extract_numeric_part)

    # Filter data to only include numeric_part values from 0.0 to 1.0
    df_sorted = df_sorted[df_sorted['numeric_part'] <= 1.0]
    for agg in aggregations:
        print(agg)
        df_current = df_sorted.copy()
        if agg == 'all':
            if per_sample:
                df_current['agg'] = df_current['model']
            else:
                df_current['agg'] = df_current['model']
        elif agg == 'model':
            df_current['agg'] = df_current['model'].apply(lambda x: '_'.join(x.split('_')[:-1]))
        elif agg == 'temp':
            df_current['agg'] = df_current['model'].apply(lambda x: x.split('_')[-1])
        elif agg == 'log':
            df_current['agg'] = df_current['model'].apply(lambda x: 'log' if 'log' in x else 'no_log')
        elif agg == 'drop':
            df_current['agg'] = df_current['model'].apply(lambda x: 'drop' if 'drop' in x else 'no_drop')
        elif agg == 'order':
            df_current['agg'] = df_current['model'].apply(lambda x: x.split('_')[0])
        elif agg == 'channels': 
            df_current['agg'] = df_current['model'].apply(lambda x: x[x.find("_c"):-1].split("_")[1])
        elif agg == 'layers':
            df_current['agg'] = df_current['model'].apply(lambda x: x.replace("_log", "")[x.replace("_log", "").find("_l"):-1].split("_")[1])
        elif agg == 'kernal':
            df_current['agg'] = df_current['model'].apply(lambda x: x[x.find("_k"):-1].split("_")[1])
        elif agg == 'architecture':
            df_current['agg'] = df_current['model'].apply(lambda x: "CFM" if "CFM" in x else "NF" if "NF" in x else "RF" if "RF" in x else "DET")
        
        
        if per_sample:
            metrics = ['distance_speaker', 'gmm', 'wv_mos', 'overlap', 'pitch_mean', 'pitch_var', 'energy_mean',
                    'energy_var', 'duration_mean', 'duration_var']
        else:
            metrics = ['distance_speaker', 'gmm', 'wv_mos', 'overlap', 'mean_sum_score', 'var_sum_score',
                    'mean_arithmetic_score', 'var_arithmetic_score', 'mean_geometric_score', 'var_geometric_score',
                    'mean_harmonic_score', 'var_harmonic_score', 'pitch_mean', 'pitch_var', 'energy_mean',
                    'energy_var', 'duration_mean', 'duration_var']
        
        for metric in metrics:
             # Normalize y-values within each group (CFM, NF, RF)
            df_current['normalized_value'] = df_current.groupby('group')[metric].transform(
                lambda x: (x - x.min()) / (x.max() - x.min())  # Min-Max Normalization
            )
            plt.figure(figsize=(15, 6))
            if metric == 'pitch_var' or metric == 'duration_var' or metric == 'pitch_mean' or metric == 'duration_mean'or metric == 'energy_mean' or metric == 'energy_var':
                df_current = df_current.sort_values(by=['sort_key', 'numeric_part'], ascending=[True, True])
                order = df_current['agg'].unique()
            else:
                order = df_current.groupby('agg')[metric].mean().sort_values(ascending=False).index
            # Create a bar plot for the metric (no grouping)
            palette_dict = {
                "NF": "#3274A1",  # Extracted blue
                "CFM": "#E1812C", # Extracted orange
                "RF": "#3A923A"   # Extracted green
            }
            # create barplot
            ax = sns.barplot(x='agg', y='normalized_value', data=df_current, hue='group', order=order, errorbar='sd', palette=palette_dict, saturation=1.3)

            # Customize plot labels and title
            plt.xlabel('Temperature', fontsize=28, weight='bold') 
            if metric == 'pitch_mean':
                y_label = "Variance of Mean"
            elif metric== 'duration_mean':
                y_label = "Variance of Mean"
            elif metric == 'pitch_var':
                y_label = "Variance of Variance"
            elif metric== 'duration_var':
                y_label = "Variance of Variance"
            elif metric== 'energy_mean':
                y_label = "Variance of Mean"
            elif metric == 'energy_var':
                y_label = "Variance of Variance"
            else:
                y_label=metric
            plt.ylabel(y_label, fontsize=28, weight='bold') 
            plt.yticks(fontsize=26)
            #plt.title(f'{metric} across Models', fontsize=16, weight='bold')

            plt.legend(fontsize=26)
            # Rotate x-axis labels for better visibility
            new_labels = [label if i % 2 == 0 else '' for i, label in enumerate(df_current['numeric_part'].astype(str))]

            # Apply the modified labels
            ax.set_xticklabels(new_labels)
            plt.xticks(fontsize=26)
            plt.tight_layout(rect=[0,0,1,1]) 

            # Save the plot
            plot_path = f"visualizations/{version}_{metric}_{agg}_plot.pdf"
            plt.savefig(plot_path)
            plot_path = f"visualizations/{version}_{metric}_{agg}_plot.png"
            plt.savefig(plot_path)
            print(f"Plot saved at: {plot_path}")
            plt.close()
        """
        plt.figure(figsize=(20, 6))
        if agg == 'drop' or agg == 'log' or agg == 'temp':
            order = df_current.groupby('group')['predicted'].mean().sort_values(ascending=False).index
            sns.barplot(x='group', y='predicted', hue='agg', order=order, data=df_current, errorbar='sd')
        else:   
            if per_sample:
                order = df_current.groupby('model')['predicted'].mean().sort_values(ascending=False).index
                sns.barplot(x='model', y='predicted', data=df_current, order=order, hue='model', errorbar='sd')
                
            elif with_variance and agg == 'all':
                ax = sns.barplot(x='model', y='predicted', data=df_current, hue='model', errorbar=None)
                # Calculate the positions of the bars
                bar_positions = range(len(df_current['agg']))
                # Manually add the error bars
                print("Predicted Values:", df_current['predicted'])
                print("Error Bars (sqrt(predicted_var)):", np.sqrt(df_current['predicted_var'].abs()))

                ax.errorbar(bar_positions, df_current['predicted'], yerr=np.sqrt(df_current['predicted_var'].abs()), fmt='none', ecolor='black', capsize=1)  
            else:
                order = df_current.groupby('agg')['predicted'].mean().sort_values(ascending=False).index
                sns.barplot(x='agg', y='predicted', hue='group', order=order, data=df_current, palette='coolwarm', errorbar='sd')
                
        # Adding labels and title
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title(path_to_predictor)
        plt.xticks(rotation=45,ha='right')

        # Adjust the plot layout to prevent cutting off labels
        plt.tight_layout(rect=[0, 0, 1, 1]) 

        plt.savefig(f"visualizations/{version}_{agg}_{path_to_predictor}_results.png")
    
        if use_wandb:
            bar_plot = wandb.Image(f"visualizations/{version}_{agg}_{path_to_predictor}_results.png")
            wandb.log({"Result barplot": bar_plot})

        plt.close()
        """

        # all metrics in one plot
        fig, axes = plt.subplots(2, 1, figsize=(30, 40))
       
        df_melted_var = pd.melt(df_current, id_vars=['agg', 'group'], value_vars=['mean_sum_score_zscore', 'var_sum_score_zscore'],
                    var_name='var', value_name='value')
        df_melted_qual = pd.melt(df_current, id_vars=['agg', 'group'], value_vars=['overlap_zscore', 'wv_mos_zscore', 'gmm_zscore', 'distance_speaker_zscore'],
                    var_name='var', value_name='value')
        
        if per_sample:
            order = df_melted_var.groupby('model')['value'].mean().sort_values(ascending=False).index
            sns.barplot(x='model', y='value', data=df_melted_var, order=order, hue='model', errorbar='sd', ax=axes[0])

            order = df_melted_qual.groupby('model')['value'].mean().sort_values(ascending=False).index
            sns.barplot(x='model', y='value', data=df_melted_qual, order=order, hue='model', errorbar='sd', ax=axes[1])
        elif agg == 'channels' or agg == 'layers' or agg == 'kernal':       
            sns.barplot(x='agg', y='value', hue='var', data=df_melted_var, palette='coolwarm', errorbar='sd', ax=axes[0])
            sns.barplot(x='agg', y='value', hue='var', data=df_melted_qual, palette='coolwarm', errorbar='sd', ax=axes[1])
    
        else:
            order = df_melted_var.groupby('agg')['value'].mean().sort_values(ascending=False).index
            sns.barplot(x='agg', y='value', hue='var', order=order, data=df_melted_var, errorbar='sd', ax=axes[0])
            order = df_melted_qual.groupby('agg')['value'].mean().sort_values(ascending=False).index
            sns.barplot(x='agg', y='value', hue='var', order=order, data=df_melted_qual, errorbar='sd', ax=axes[1])
    
        # Adding labels and title
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title("Metrics")
        plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right")
        plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")

        # Adjust the plot layout to prevent cutting off labels
        plt.tight_layout(rect=[0, 0, 1, 1]) 
        plt.savefig(f"visualizations/{version}_{agg}_metrics_results.pdf")
        
        if use_wandb:
            bar_plot = wandb.Image(f"visualizations/{version}_{agg}_metrics_results.png")
            wandb.log({"Metrics barplot": bar_plot})



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Evaluation')

    parser.add_argument('--model_dir',
                        type=str,
                        help="Path to Models.",
                        default=None)
    
    parser.add_argument('--version',
                        type=str,
                        help="Identifies the Run.",
                        default=None)
    
    parser.add_argument('--gpu_id',
                        type=str,
                        help="Which GPU(s) to run on. If not specified runs on CPU, but other than for integration tests that doesn't make much sense.",
                        default="cpu")

    parser.add_argument('--wandb',
                        action="store_true",
                        help="Whether to use weights and biases to track training runs. Requires you to run wandb login and place your auth key before.",
                        default=False)
    parser.add_argument('--per_sample',
                        action="store_true",
                        help="Whether to compute MOS and Variance per sample or per model.",
                        default=False)
    parser.add_argument('--multi_speaker',
                        action="store_true",
                        help="Whether to use more than one speaker.",
                        default=False)

    args = parser.parse_args()

    run_eval(path_to_models = args.model_dir, version=args.version, use_wandb = args.wandb, gpu_id = args.gpu_id, per_sample=args.per_sample, multi_speaker=args.multi_speaker)
