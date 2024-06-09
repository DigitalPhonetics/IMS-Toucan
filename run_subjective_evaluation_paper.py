import os
from statistics import mean
import math

from Evaluation.subjective_evaluation import *
from Evaluation.plotting import *
from Utility.storage_config import PREPROCESSING_DIR

import sys

if __name__ == '__main__':
    os.makedirs(os.path.join(PREPROCESSING_DIR, "Evaluation", "plots_paper"), exist_ok=True)
    save_dir = os.path.join(PREPROCESSING_DIR, "Evaluation", "plots_paper")

    # load data
    data = read_data(os.path.join(PREPROCESSING_DIR, "Evaluation", "data_listeningtestmaster_2023-07-30_16-07.csv"))

    sd = sociodemographics(data)

    pref = preference(data)

    sim = similarity(data)

    mos_original = mean_opinion_score(data, "O")
    mos_baseline = mean_opinion_score(data, "B")
    mos_proposed = mean_opinion_score(data, "S")
    mos_prompt = mean_opinion_score(data, "P")

    #print(sum(sum(inner_dict.values()) for inner_dict in mos_original.values()))
    #print(sum(sum(inner_dict.values()) for inner_dict in mos_baseline.values()))
    #print(sum(sum(inner_dict.values()) for inner_dict in mos_proposed.values()))
    #print(sum(sum(inner_dict.values()) for inner_dict in mos_prompt.values()))

    emotion_original = emotion(data, "O")
    emotion_baseline = emotion(data, "B")
    emotion_proposed = emotion(data, "S")
    emotion_prompt = emotion(data, "P")

    valence_original = valence(data, "O")
    valence_baseline = valence(data, "B")
    valence_proposed = valence(data, "S")
    valence_prompt = valence(data, "P")

    arousal_original = arousal(data, "O")
    arousal_baseline = arousal(data, "B")
    arousal_proposed = arousal(data, "S")
    arousal_prompt = arousal(data, "P")

    # transform data
    #################
    #################

    # A/B pref
    ##############
    pref_female, pref_male = split_female_male(pref)
    pref_female_total = collapse_subdicts(pref_female)
    pref_male_total = collapse_subdicts(pref_male)
    pref_total = {}
    for key, value in pref_female_total.items():
        if key not in pref_total:
            pref_total[key] = 0
        pref_total[key] += value
    for key, value in pref_male_total.items():
        if key not in pref_total:
            pref_total[key] = 0
        pref_total[key] += value

    #print(pref_total)

    # similarity
    #############
    sim_female, sim_male = split_female_male(sim)
    msim_female = get_mean_rating_nested(remove_outliers(sim_female))
    msim_male = get_mean_rating_nested(remove_outliers(sim_male))
    msim_total = {'female' : mean(list(msim_female.values())),
                  'male' : mean(list(msim_male.values()))}
    
    sim_female = remove_outliers(sim_female)
    sim_male = remove_outliers(sim_male)
    print(sim_male)

    def calculate_stats(data):
        emotions = {}
        total_count = 0
        total_sum = 0
        for emotion, ratings in data.items():
            total_sum = 0
            squared_deviations = 0
            total_count_emotion = sum(ratings.values())
            for rating, count in ratings.items():
                total_sum += rating * count
                squared_deviations += (rating - (total_sum / total_count_emotion))**2 * count
            mean = total_sum / total_count_emotion
            std_dev = math.sqrt(squared_deviations / (total_count_emotion - 1))
            emotions[emotion] = {"mean": mean, "std_dev": std_dev}
            total_count += total_count_emotion
            total_sum += total_sum

        overall_mean = total_sum / total_count
        overall_squared_deviations = 0
        for emotion, stats in emotions.items():
            emotion_mean = stats["mean"]
        for rating, count in data[emotion].items():
            overall_squared_deviations += (rating - overall_mean) ** 2 * count
        overall_std_dev = math.sqrt(overall_squared_deviations / (total_count - 1))

        return {"emotions": emotions, "overall_mean": overall_mean, "overall_std_dev": overall_std_dev}
    
    # Calculate the total count
    total_count = 0
    for e, ratings in sim_female.items():
        total_count += sum(ratings.values())
    print(total_count)
    # Calculate the total count
    total_count = 0
    for e, ratings in sim_male.items():
        total_count += sum(ratings.values())
    print(total_count)
    
    print(calculate_stats(sim_female))

    # mos
    ##############
    mos_original_female, mos_original_male = split_female_male(mos_original)
    mos_baseline_female, mos_baseline_male = split_female_male(mos_baseline)
    mos_proposed_female, mos_proposed_male = split_female_male(mos_proposed)
    mos_prompt_female, mos_prompt_male = split_female_male(mos_prompt)
    # combine proposed and prompt since it is the same system and the distinction is not really needed for mos
    mos_proposed_female = combine_dicts(mos_proposed_female, mos_prompt_female)
    mos_proposed_male = combine_dicts(mos_proposed_male, mos_prompt_male)

    omos_original_female = get_mean_rating_nested(remove_outliers(mos_original_female))
    omos_original_male = get_mean_rating_nested(remove_outliers(mos_original_male))
    omos_baseline_female = get_mean_rating_nested(remove_outliers(mos_baseline_female))
    omos_baseline_male = get_mean_rating_nested(remove_outliers(mos_baseline_male))
    omos_proposed_female = get_mean_rating_nested(remove_outliers(mos_proposed_female))
    omos_proposed_male = get_mean_rating_nested(remove_outliers(mos_proposed_male))

    omos_female = [mean(list(omos_original_female.values())), mean(list(omos_baseline_female.values())), mean(list(omos_proposed_female.values()))]
    omos_male = [mean(list(omos_original_male.values())), mean(list(omos_baseline_male.values())), mean(list(omos_proposed_male.values()))]
    omos_all = [mean([m1, m2]) for m1, m2 in zip(omos_female, omos_male)]

    print(omos_female)

    _, p_value_mos = independent_samples_t_test(mos_proposed_female, mos_proposed_male, mos_baseline_female, mos_baseline_male)
    print(f'p value MOS proposed-baseline: {p_value_mos}')

    # emotion
    ###########
    emotion_original_female, emotion_original_male = split_female_male(emotion_original)
    emotion_baseline_female, emotion_baseline_male = split_female_male(emotion_baseline)
    emotion_proposed_female, emotion_proposed_male = split_female_male(emotion_proposed)
    emotion_prompt_female, emotion_prompt_male = split_female_male(emotion_prompt)
    emotion_prompt_female = make_emotion_prompts(emotion_prompt_female, "f")
    emotion_prompt_male = make_emotion_prompts(emotion_prompt_male, "m")

    print(cramers_v(emotion_original_female))
    print(cramers_v(emotion_baseline_female))
    print(cramers_v(emotion_proposed_female))
    print(cramers_v(emotion_prompt_female))
    print(cramers_v(emotion_original_male))
    print(cramers_v(emotion_baseline_male))
    print(cramers_v(emotion_proposed_male))
    print(cramers_v(emotion_prompt_male))

    # valence/arousal
    ###################
    valence_original_female, valence_original_male = split_female_male(valence_original)
    arousal_original_female, arousal_original_male = split_female_male(arousal_original)
    valence_baseline_female, valence_baseline_male = split_female_male(valence_baseline)
    arousal_baseline_female, arousal_baseline_male = split_female_male(arousal_baseline)
    valence_proposed_female, valence_proposed_male = split_female_male(valence_proposed)
    arousal_proposed_female, arousal_proposed_male = split_female_male(arousal_proposed)
    valence_prompt_female, valence_prompt_male = split_female_male(valence_prompt)
    arousal_prompt_female, arousal_prompt_male = split_female_male(arousal_prompt)
    valence_prompt_female = make_emotion_prompts(valence_prompt_female, "f")
    valence_prompt_male = make_emotion_prompts(valence_prompt_male, "m")
    arousal_prompt_female = make_emotion_prompts(arousal_prompt_female, "f")
    arousal_prompt_male = make_emotion_prompts(arousal_prompt_male, "m")

    mvalence_original_female = get_mean_rating_nested(remove_outliers(valence_original_female))
    mvalence_original_male = get_mean_rating_nested(remove_outliers(valence_original_male))
    mvalence_baseline_female = get_mean_rating_nested(remove_outliers(valence_baseline_female))
    mvalence_baseline_male = get_mean_rating_nested(remove_outliers(valence_baseline_male))
    mvalence_proposed_female = get_mean_rating_nested(remove_outliers(valence_proposed_female))
    mvalence_proposed_male = get_mean_rating_nested(remove_outliers(valence_proposed_male))
    mvalence_prompt_female = get_mean_rating_nested(remove_outliers(valence_prompt_female))
    mvalence_prompt_male = get_mean_rating_nested(remove_outliers(valence_prompt_male))

    marousal_original_female = get_mean_rating_nested(remove_outliers(arousal_original_female))
    marousal_original_male = get_mean_rating_nested(remove_outliers(arousal_original_male))
    marousal_baseline_female = get_mean_rating_nested(remove_outliers(arousal_baseline_female))
    marousal_baseline_male = get_mean_rating_nested(remove_outliers(arousal_baseline_male))
    marousal_proposed_female = get_mean_rating_nested(remove_outliers(arousal_proposed_female))
    marousal_proposed_male = get_mean_rating_nested(remove_outliers(arousal_proposed_male))
    marousal_prompt_female = get_mean_rating_nested(remove_outliers(arousal_prompt_female))
    marousal_prompt_male = get_mean_rating_nested(remove_outliers(arousal_prompt_male))

    print(marousal_original_female)
    print(marousal_original_male)
    
    print(marousal_baseline_female)
    print(marousal_baseline_male)

    print(marousal_proposed_female)
    print(marousal_proposed_male)

    print(marousal_prompt_female)
    print(marousal_prompt_male)


    # make plots
    #################
    #################

    pie_barplot_pref_total(pref_total, os.path.join(save_dir, f"pref_total_pie.pdf"))

    pie_barplot_pref(pref_female, os.path.join(save_dir, f"pref_pie.pdf"))
