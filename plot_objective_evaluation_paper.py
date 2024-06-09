import os
from statistics import median, mean
import numpy as np
import scipy.stats as stats

import torch

from Utility.storage_config import PREPROCESSING_DIR
from Evaluation.objective_evaluation import *
from Evaluation.plotting import *

import sys

EMOTIONS = ["anger", "joy", "neutral", "sadness", "surprise"]

def get_ratings_per_speaker(data):
    ratings = {}
    for dataset, speakers in data.items():
        for speaker, emotions in speakers.items():
            if speaker not in ratings:
                ratings[speaker] = []
            ratings[speaker].extend(list(emotions.values()))
    return ratings

def get_ratings_per_speaker_original(data):
    ratings = {}
    for speaker, emotions in data.items():
        ratings[speaker] = list(emotions.values())
    return ratings

def get_single_rating_per_speaker(data):
    rating = {}
    for speaker, ratings in data.items():
        rating[speaker] = mean(ratings)
    return rating

def remove_outliers_per_speaker(data):
    # data shape: {speaker: {ratings}}
    cleaned_data = {}
    for speaker, ratings_list in data.items():
        sorted_data = sorted(ratings_list)
        q1, q3 = np.percentile(sorted_data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        cleaned_data[speaker] = [x for x in sorted_data if lower_bound <= x <= upper_bound]
    return cleaned_data

def get_ratings_per_emotion(data):
    ratings = {}
    for dataset, speakers in data.items():
        for speaker, emotions in speakers.items():
            for emotion, preds in emotions.items():
                if emotion not in ratings:
                    ratings[emotion] = {}
                for pred, freq in preds.items():
                    if pred not in ratings[emotion]:
                        ratings[emotion][pred] = 0
                    ratings[emotion][pred] += freq
    return ratings

def get_ratings_per_emotion_original(data):
    ratings = {}
    for speaker, emotions in data.items():
        for emotion, preds in emotions.items():
            if emotion not in ratings:
                ratings[emotion] = {}
            for pred, freq in preds.items():
                if pred not in ratings[emotion]:
                    ratings[emotion][pred] = 0
                ratings[emotion][pred] += freq
    return ratings

def get_ratings_per_speaker_emotion(data, speaker_ids):
    ratings = {}
    for dataset, speakers in data.items():
        for speaker, emotions in speakers.items():
            if int(speaker) in speaker_ids:
                if speaker not in ratings:
                    ratings[speaker] = {}
                for emotion, preds in emotions.items():
                    if emotion not in ratings[speaker]:
                        ratings[speaker][emotion] = {}
                    for pred, freq in preds.items():
                        if pred not in ratings[speaker][emotion]:
                            ratings[speaker][emotion][pred] = 0
                        ratings[speaker][emotion][pred] += freq
    return ratings

def get_ratings_per_speaker_emotion_original(data, speaker_ids):
    ratings = {}
    for speaker, emotions in data.items():
        if int(speaker) in speaker_ids:
            ratings[speaker] = {}
            for emotion, preds in emotions.items():
                if emotion not in ratings[speaker]:
                    ratings[speaker][emotion] = {}
                for pred, freq in preds.items():
                    if pred not in ratings[speaker][emotion]:
                        ratings[speaker][emotion][pred] = 0
                    ratings[speaker][emotion][pred] += freq
    return ratings

def total_accuracy(data):
    count_correct = 0
    count_total = 0
    for emotion, preds in data.items():
        for pred, freq in preds.items():
            if pred == emotion:
                count_correct += freq
            count_total += freq
    return count_correct / count_total

def combine_sent_prompt(dict1, dict2):
    combined_dict = {}
    for key in dict1.keys() | dict2.keys():
        combined_dict[key] = dict1[key] + dict2[key]
    return combined_dict

def get_dict_with_rounded_values(dict, decimal_points=3):
    rounded_dict = {key: round(value, decimal_points) for key, value in dict.items()}
    return rounded_dict

def cramers_v(data):
    # Convert the data dictionary into a 2D array
    counts = np.array([[data[emotion].get(label, 0) for emotion in EMOTIONS] for label in EMOTIONS])

    # Compute the chi-squared statistic and p-value
    chi2, p, _, _ = stats.chi2_contingency(counts)

    # Number of observations (total counts)
    n = np.sum(counts)

    # Number of rows and columns in the contingency table
    num_rows = len(EMOTIONS)
    num_cols = len(EMOTIONS)

    # Compute Cramér's V
    cramer_v = np.sqrt(chi2 / (n * (min(num_rows, num_cols) - 1)))
    return p, cramer_v

if __name__ == '__main__':
    # load results

    # emotion recognition
    # shape {dataset: {speaker: {emotion: {pred_emotion: count}}}}
    freqs_original = torch.load(os.path.join(PREPROCESSING_DIR, "Evaluation", "freqs_original.pt"), map_location='cpu')
    freqs_baseline = torch.load(os.path.join(PREPROCESSING_DIR, "Evaluation", "freqs_baseline.pt"), map_location='cpu')
    freqs_sent = torch.load(os.path.join(PREPROCESSING_DIR, "Evaluation", "freqs_sent.pt"), map_location='cpu')
    freqs_prompt = torch.load(os.path.join(PREPROCESSING_DIR, "Evaluation", "freqs_prompt.pt"), map_location='cpu')
    freqs_emospeech = torch.load("/mount/arbeitsdaten/synthesis/bottts/emospeech/Evaluation/freqs.pt", map_location='cpu')

    speaker_similarities_emospeech = torch.load("/mount/arbeitsdaten/synthesis/bottts/emospeech/Evaluation/speaker_similarities.pt", map_location='cpu')

    # Extracting all the scores
    all_scores = []
    for key in speaker_similarities_emospeech:
        for sub_key in speaker_similarities_emospeech[key]:
            all_scores.extend(list(speaker_similarities_emospeech[key][sub_key].values()))

    # Computing mean and standard deviation
    mean_all = np.mean(all_scores)
    std_dev_all = np.std(all_scores)
    print(mean_all)
    print(std_dev_all)



    # extract ratings

    # emotion recognition

    # per emotion
    freqs_original_emotion = get_ratings_per_emotion_original(freqs_original)
    freqs_baseline_emotion = get_ratings_per_emotion(freqs_baseline)
    freqs_sent_emotion = get_ratings_per_emotion(freqs_sent)
    freqs_prompt_emotion = get_ratings_per_emotion(freqs_prompt)
    freqs_emospeech_emotion = get_ratings_per_emotion(freqs_emospeech)

    # plotting
    os.makedirs(os.path.join(PREPROCESSING_DIR, "Evaluation", "plots_paper"), exist_ok=True)
    save_dir = os.path.join(PREPROCESSING_DIR, "Evaluation", "plots_paper")

    #heatmap_emotion(freqs_original_emotion, os.path.join(save_dir, f"emotion_objective_original.png"))
    #heatmap_emotion(freqs_baseline_emotion, os.path.join(save_dir, f"emotion_objective_baseline.png"))
    #heatmap_emotion(freqs_sent_emotion, os.path.join(save_dir, f"emotion_objective_sent.png"))
    #heatmap_emotion(freqs_prompt_emotion, os.path.join(save_dir, f"emotion_objective_prompt.png"))
    titles = ["Ground Truth", "Baseline", "Prompt Conditioned Same", "Prompt Conditioned Other", "EmoSpeech"]
    heatmap_emotion_multiple([freqs_original_emotion, freqs_baseline_emotion, freqs_sent_emotion, freqs_prompt_emotion, freqs_emospeech_emotion], titles, os.path.join(save_dir, f"emotion_objective_all2.pdf"))

    print("Cramers V")
    data1 = freqs_original_emotion
    data2 = freqs_emospeech_emotion
    print(freqs_emospeech_emotion)
    _, v1 = cramers_v(data1)
    _, v2 = cramers_v(data2)
    print(v1)
    print(v2)

    # Compute the standard error of the difference
    n1 = sum(sum(emotion.values()) for emotion in data1.values())
    n2 = sum(sum(emotion.values()) for emotion in data2.values())
    se_diff = np.sqrt((v1**2 / (n1 - 1)) + (v2**2 / (n2 - 1)))

    # Compute the t-statistic
    t_statistic = (v1 - v2) / se_diff

    # Compute the degrees of freedom
    df = min(len(data1) - 1, len(data2) - 1)

    # Compute the p-value
    p_value = stats.t.cdf(t_statistic, df)
    print(t_statistic)
    print(p_value)

    sys.exit()
    accuracies_emotion_original = {} # per speaker per emotion
    accuracies_speaker_original = {} # per speaker
    for speaker, emotions in freqs_original.items():
        accuracies_emotion_original[speaker] = {}
        accuracies_speaker_original[speaker] = sum([emotions[emo][pred] 
                                                    for emo, preds in emotions.items() 
                                                    for pred in preds if pred == emo]) / sum([emotions[emo][pred] 
                                                                                              for emo, preds in emotions.items() 
                                                                                              for pred in preds])
        for emotion, pred_emotions in emotions.items():
            accuracies_emotion_original[speaker][emotion] = pred_emotions[emotion] / sum(list(pred_emotions.values()))

    accuracy_original = sum([freqs_original[speaker][emotion][pred]
                                for speaker, emotions in freqs_original.items()
                                for emotion, preds in emotions.items() 
                                for pred in preds if pred == emotion]) / sum([freqs_original[speaker][emotion][pred]
                                                                                for speaker, emotions in freqs_original.items()
                                                                                for emotion, preds in emotions.items() 
                                                                                for pred in preds])
    
    accuracies_emotion_baseline = {} # per dataset per speaker per emotion
    accuracies_speaker_baseline = {} # per speaker
    count_correct = {}
    count_total = {}
    for dataset, speakers in freqs_baseline.items():
        accuracies_emotion_baseline[dataset] = {}
        for speaker, emotions in speakers:
            accuracies_emotion_baseline[dataset][speaker] = {}
            if speaker not in count_correct:
                count_correct[speaker] = 0
            if speaker not in count_total:
                count_total[speaker] = 0
            count_correct[speaker] += sum([emotions[emo][pred] 
                                                        for emo, preds in emotions.items() 
                                                        for pred in preds if pred == emo])
            count_total[speaker] += sum([emotions[emo][pred] 
                                                    for emo, preds in emotions.items() 
                                                    for pred in preds])
            for emotion, pred_emotions in emotions.items():
                accuracies_emotion_baseline[dataset][speaker][emotion] = pred_emotions[emotion] / sum(list(pred_emotions.values()))
    for speaker, freq in count_correct.items():
        accuracies_speaker_baseline[speaker] = freq / count_total[speaker]

    accuracy_baseline = sum([freqs_baseline[dataset][speaker][emotion][pred]
                                for dataset, speakers in freqs_baseline.items()
                                for speaker, emotions in speakers
                                for emotion, preds in emotions.items() 
                                for pred in preds if pred == emotion]) / sum([freqs_baseline[dataset][speaker][emotion][pred]
                                                                                for dataset, speakers in freqs_baseline.items()
                                                                                for speaker, emotions in speakers
                                                                                for emotion, preds in emotions.items() 
                                                                                for pred in preds])
    
    accuracies_emotion_sent = {} # per dataset per speaker per emotion
    accuracies_speaker_sent = {} # per speaker
    count_correct = {}
    count_total = {}
    for dataset, speakers in freqs_sent.items():
        accuracies_emotion_sent[dataset] = {}
        for speaker, emotions in speakers:
            accuracies_emotion_sent[dataset][speaker] = {}
            if speaker not in count_correct:
                count_correct[speaker] = 0
            if speaker not in count_total:
                count_total[speaker] = 0
            count_correct[speaker] += sum([emotions[emo][pred] 
                                                        for emo, preds in emotions.items() 
                                                        for pred in preds if pred == emo])
            count_total[speaker] += sum([emotions[emo][pred] 
                                                    for emo, preds in emotions.items() 
                                                    for pred in preds])
            for emotion, pred_emotions in emotions.items():
                accuracies_emotion_sent[dataset][speaker][emotion] = pred_emotions[emotion] / sum(list(pred_emotions.values()))
    for speaker, freq in count_correct.items():
        accuracies_speaker_sent[speaker] = freq / count_total[speaker]

    accuracy_sent = sum([freqs_sent[dataset][speaker][emotion][pred]
                                for dataset, speakers in freqs_sent.items()
                                for speaker, emotions in speakers
                                for emotion, preds in emotions.items() 
                                for pred in preds if pred == emotion]) / sum([freqs_sent[dataset][speaker][emotion][pred]
                                                                                for dataset, speakers in freqs_sent.items()
                                                                                for speaker, emotions in speakers
                                                                                for emotion, preds in emotions.items() 
                                                                                for pred in preds])
    
    accuracies_emotion_prompt = {} # per dataset per speaker per emotion
    accuracies_speaker_prompt = {} # per speaker
    count_correct = {}
    count_total = {}
    for dataset, speakers in freqs_prompt.items():
        accuracies_emotion_prompt[dataset] = {}
        for speaker, emotions in speakers:
            accuracies_emotion_prompt[dataset][speaker] = {}
            if speaker not in count_correct:
                count_correct[speaker] = 0
            if speaker not in count_total:
                count_total[speaker] = 0
            count_correct[speaker] += sum([emotions[emo][pred] 
                                                        for emo, preds in emotions.items() 
                                                        for pred in preds if pred == emo])
            count_total[speaker] += sum([emotions[emo][pred] 
                                                    for emo, preds in emotions.items() 
                                                    for pred in preds])
            for emotion, pred_emotions in emotions.items():
                accuracies_emotion_prompt[dataset][speaker][emotion] = pred_emotions[emotion] / sum(list(pred_emotions.values()))
    for speaker, freq in count_correct.items():
        accuracies_speaker_prompt[speaker] = freq / count_total[speaker]

    accuracy_prompt = sum([freqs_prompt[dataset][speaker][emotion][pred]
                                for dataset, speakers in freqs_prompt.items()
                                for speaker, emotions in speakers
                                for emotion, preds in emotions.items() 
                                for pred in preds if pred == emotion]) / sum([freqs_prompt[dataset][speaker][emotion][pred]
                                                                                for dataset, speakers in freqs_prompt.items()
                                                                                for speaker, emotions in speakers
                                                                                for emotion, preds in emotions.items() 
                                                                                for pred in preds])