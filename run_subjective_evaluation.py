import os
from statistics import mean

from Evaluation.subjective_evaluation import *
from Evaluation.plotting import *
from Utility.storage_config import PREPROCESSING_DIR

import sys

if __name__ == '__main__':
    os.makedirs(os.path.join(PREPROCESSING_DIR, "Evaluation", "plots"), exist_ok=True)
    save_dir = os.path.join(PREPROCESSING_DIR, "Evaluation", "plots")

    # load data
    data = read_data(os.path.join(PREPROCESSING_DIR, "Evaluation", "data_listeningtestmaster_2023-07-03_13-06.csv"))

    sd = sociodemographics(data)

    pref = preference(data)

    sim = similarity(data)

    mos_original = mean_opinion_score(data, "O")
    mos_baseline = mean_opinion_score(data, "B")
    mos_proposed = mean_opinion_score(data, "S")
    mos_prompt = mean_opinion_score(data, "P")

    emotion_original = emotion(data, "O")
    emotion_baseline = emotion(data, "B")
    emotion_proposed = emotion(data, "S")
    emotion_prompt = emotion(data, "P")

    va_original = valence_arousal(data, "O")
    va_baseline = valence_arousal(data, "B")
    va_proposed = valence_arousal(data, "S")
    va_prompt = valence_arousal(data, "P")

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

    # similarity
    #############
    sim_female, sim_male = split_female_male(sim)
    msim_female = get_mean_rating_nested(remove_outliers(sim_female))
    msim_male = get_mean_rating_nested(remove_outliers(sim_male))
    msim_total = {'female' : mean(list(msim_female.values())),
                  'male' : mean(list(msim_male.values()))}

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

    # emotion
    ###########
    emotion_original_female, emotion_original_male = split_female_male(emotion_original)
    emotion_baseline_female, emotion_baseline_male = split_female_male(emotion_baseline)
    emotion_proposed_female, emotion_proposed_male = split_female_male(emotion_proposed)
    emotion_prompt_female, emotion_prompt_male = split_female_male(emotion_prompt)
    emotion_prompt_female = make_emotion_prompts(emotion_prompt_female, "f")
    emotion_prompt_male = make_emotion_prompts(emotion_prompt_male, "m")

    # valence/arousal
    ###################
    va_original_female = {}
    va_original_male = {}
    va_original_female["valence"], va_original_male["valence"] = split_female_male(va_original["valence"])
    va_original_female["arousal"], va_original_male["arousal"] = split_female_male(va_original["arousal"])
    va_baseline_female = {}
    va_baseline_male = {}
    va_baseline_female["valence"], va_baseline_male["valence"] = split_female_male(va_baseline["valence"])
    va_baseline_female["arousal"], va_baseline_male["arousal"] = split_female_male(va_baseline["arousal"])
    va_proposed_female = {}
    va_proposed_male = {}
    va_proposed_female["valence"], va_proposed_male["valence"] = split_female_male(va_proposed["valence"])
    va_proposed_female["arousal"], va_proposed_male["arousal"] = split_female_male(va_proposed["arousal"])
    va_prompt_female = {}
    va_prompt_male = {}
    va_prompt_female["valence"], va_prompt_male["valence"] = split_female_male(va_prompt["valence"])
    va_prompt_female["arousal"], va_prompt_male["arousal"] = split_female_male(va_prompt["arousal"])
    va_prompt_female["valence"] = make_emotion_prompts(va_prompt_female["valence"], "f")
    va_prompt_female["arousal"] = make_emotion_prompts(va_prompt_female["arousal"], "f")
    va_prompt_male["valence"] = make_emotion_prompts(va_prompt_male["valence"], "m")
    va_prompt_male["arousal"] = make_emotion_prompts(va_prompt_male["arousal"], "m")


    # make plots
    #################
    #################

    for v, d in sd.items():
        pie_chart_counts(d, v, os.path.join(save_dir, f"{v}.png"))

    barplot_pref3(pref_female, os.path.join(save_dir, f"pref_female.png"))
    barplot_pref3(pref_male, os.path.join(save_dir, f"pref_male.png"))
    barplot_pref_total(pref_female_total, os.path.join(save_dir, f"pref_female_total.png"))
    barplot_pref_total(pref_male_total, os.path.join(save_dir, f"pref_male_total.png"))
    barplot_pref_total(pref_total, os.path.join(save_dir, f"pref_total.png"))

    boxplot_rating(sim_female, os.path.join(save_dir, f"box_sim_female.png"))
    boxplot_rating(sim_male, os.path.join(save_dir, f"box_sim_male.png"))
    barplot_sim(msim_female, os.path.join(save_dir, f"sim_female.png"))
    barplot_sim(msim_male, os.path.join(save_dir, f"sim_male.png"))
    barplot_sim_total(msim_total, os.path.join(save_dir, f"sim_total.png"))

    boxplot_rating(mos_original_female, os.path.join(save_dir, f"box_mos_original_female.png"))
    boxplot_rating(mos_original_male, os.path.join(save_dir, f"box_mos_original_male.png"))
    boxplot_rating(mos_baseline_female, os.path.join(save_dir, f"box_mos_baseline_female.png"))
    boxplot_rating(mos_baseline_male, os.path.join(save_dir, f"box_mos_baseline_male.png"))
    boxplot_rating(mos_proposed_female, os.path.join(save_dir, f"box_mos_proposed_female.png"))
    boxplot_rating(mos_proposed_male, os.path.join(save_dir, f"box_mos_proposed_male.png"))
    barplot_mos(omos_female, os.path.join(save_dir, f"mos_female.png"))
    barplot_mos(omos_male, os.path.join(save_dir, f"mos_male.png"))
    barplot_mos(omos_all, os.path.join(save_dir, f"mos.png"))

    barplot_emotion(emotion_original_female, os.path.join(save_dir, f"emotion_original_female.png"))
    barplot_emotion(emotion_original_male, os.path.join(save_dir, f"emotion_original_male.png"))
    barplot_emotion(emotion_baseline_female, os.path.join(save_dir, f"emotion_baseline_female.png"))
    barplot_emotion(emotion_baseline_male, os.path.join(save_dir, f"emotion_baseline_male.png"))
    barplot_emotion(emotion_proposed_female, os.path.join(save_dir, f"emotion_proposed_female.png"))
    barplot_emotion(emotion_proposed_male, os.path.join(save_dir, f"emotion_proposed_male.png"))
    barplot_emotion(emotion_prompt_female, os.path.join(save_dir, f"emotion_prompt_female.png"))
    barplot_emotion(emotion_prompt_male, os.path.join(save_dir, f"emotion_prompt_male.png"))

    scatterplot_va(va_original_female, os.path.join(save_dir, f"va_original_female.png"))
    scatterplot_va(va_original_male, os.path.join(save_dir, f"va_original_male.png"))
    scatterplot_va(va_baseline_female, os.path.join(save_dir, f"va_baseline_female.png"))
    scatterplot_va(va_baseline_male, os.path.join(save_dir, f"va_baseline_male.png"))
    scatterplot_va(va_proposed_female, os.path.join(save_dir, f"va_proposed_female.png"))
    scatterplot_va(va_proposed_male, os.path.join(save_dir, f"va_proposed_male.png"))
    scatterplot_va(va_prompt_female, os.path.join(save_dir, f"va_prompt_female.png"))
    scatterplot_va(va_prompt_male, os.path.join(save_dir, f"va_prompt_male.png"))
