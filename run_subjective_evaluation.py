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
    data = read_data(os.path.join(PREPROCESSING_DIR, "Evaluation", "data_listeningtestmaster_2023-07-30_16-07.csv"))

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

    heatmap_emotion(emotion_original_female, os.path.join(save_dir, f"emotion_original_female.png"))
    heatmap_emotion(emotion_original_male, os.path.join(save_dir, f"emotion_original_male.png"))
    heatmap_emotion(emotion_baseline_female, os.path.join(save_dir, f"emotion_baseline_female.png"))
    heatmap_emotion(emotion_baseline_male, os.path.join(save_dir, f"emotion_baseline_male.png"))
    heatmap_emotion(emotion_proposed_female, os.path.join(save_dir, f"emotion_proposed_female.png"))
    heatmap_emotion(emotion_proposed_male, os.path.join(save_dir, f"emotion_proposed_male.png"))
    heatmap_emotion(emotion_prompt_female, os.path.join(save_dir, f"emotion_prompt_female.png"))
    heatmap_emotion(emotion_prompt_male, os.path.join(save_dir, f"emotion_prompt_male.png"))

    boxplot_rating(valence_original_female, os.path.join(save_dir, f"box_v_original_female.png"))
    boxplot_rating(valence_original_male, os.path.join(save_dir, f"box_v_original_male.png"))
    boxplot_rating(valence_baseline_female, os.path.join(save_dir, f"box_v_baseline_female.png"))
    boxplot_rating(valence_baseline_male, os.path.join(save_dir, f"box_v_baseline_male.png"))
    boxplot_rating(valence_proposed_female, os.path.join(save_dir, f"box_v_proposed_female.png"))
    boxplot_rating(valence_proposed_male, os.path.join(save_dir, f"box_v_proposed_male.png"))
    boxplot_rating(valence_prompt_female, os.path.join(save_dir, f"box_v_prompt_female.png"))
    boxplot_rating(valence_prompt_male, os.path.join(save_dir, f"box_v_prompt_male.png"))

    boxplot_rating(arousal_original_female, os.path.join(save_dir, f"box_a_original_female.png"))
    boxplot_rating(arousal_original_male, os.path.join(save_dir, f"box_a_original_male.png"))
    boxplot_rating(arousal_baseline_female, os.path.join(save_dir, f"box_a_baseline_female.png"))
    boxplot_rating(arousal_baseline_male, os.path.join(save_dir, f"box_a_baseline_male.png"))
    boxplot_rating(arousal_proposed_female, os.path.join(save_dir, f"box_a_proposed_female.png"))
    boxplot_rating(arousal_proposed_male, os.path.join(save_dir, f"box_a_proposed_male.png"))
    boxplot_rating(arousal_prompt_female, os.path.join(save_dir, f"box_a_prompt_female.png"))
    boxplot_rating(arousal_prompt_male, os.path.join(save_dir, f"box_a_prompt_male.png"))

    scatterplot_va(mvalence_original_female, marousal_original_female, os.path.join(save_dir, f"va_original_female.png"))
    scatterplot_va(mvalence_original_male, marousal_original_male, os.path.join(save_dir, f"va_original_male.png"))
    scatterplot_va(mvalence_baseline_female, marousal_baseline_female, os.path.join(save_dir, f"va_baseline_female.png"))
    scatterplot_va(mvalence_baseline_male, marousal_baseline_male, os.path.join(save_dir, f"va_baseline_male.png"))
    scatterplot_va(mvalence_proposed_female, marousal_proposed_female, os.path.join(save_dir, f"va_proposed_female.png"))
    scatterplot_va(mvalence_proposed_male, marousal_proposed_male, os.path.join(save_dir, f"va_proposed_male.png"))
    scatterplot_va(mvalence_prompt_female, marousal_prompt_female, os.path.join(save_dir, f"va_prompt_female.png"))
    scatterplot_va(mvalence_prompt_male, marousal_prompt_male, os.path.join(save_dir, f"va_prompt_male.png"))
