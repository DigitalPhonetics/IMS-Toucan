import os

from Evaluation.subjective_evaluation import *
from Utility.storage_config import PREPROCESSING_DIR

if __name__ == '__main__':
    os.makedirs(os.path.join(PREPROCESSING_DIR, "Evaluation"), exist_ok=True)
    os.makedirs(os.path.join(PREPROCESSING_DIR, "Evaluation", "plots"), exist_ok=True)
    save_dir = os.path.join(PREPROCESSING_DIR, "Evaluation", "plots")

    data = read_data("/mount/arbeitsdaten/synthesis/bottts/IMS-Toucan/Corpora/Evaluation/data_listeningtestmaster_2023-06-27_14-49.csv")
    sd = sociodemographics(data)
    pref = preference(data)
    sim = similarity(data)
    sim_mean = get_mean_rating_nested(sim)
    mos_original = mean_opinion_score(data, "O")
    mos_baseline = mean_opinion_score(data, "B")
    mos_proposed = mean_opinion_score(data, "S")
    mos_prompt = mean_opinion_score(data, "P")
    mos = {"original": mean(list(mos_original.values())),
           "baseline": mean(list(mos_baseline.values())),
           "proposed": mean(list(mos_proposed.values())),
           "prompt": mean(list(mos_prompt.values()))}
    emotion_original = emotion(data, "O")
    emotion_baseline = emotion(data, "B")
    emotion_proposed = emotion(data, "S")
    emotion_prompt = emotion(data, "P")
    va_original = valence_arousal(data, "O")
    va_baseline = valence_arousal(data, "B")
    va_proposed = valence_arousal(data, "S")
    va_prompt = valence_arousal(data, "P")

    for v, d in sd.items():
        barplot_counts(d, v, os.path.join(save_dir, f"{v}.png"))
    for v, d in pref.items():
        barplot_counts(d, v, os.path.join(save_dir, f"{v}.png"))
    for v, d in sim.items():
        barplot_counts(d, v, os.path.join(save_dir, f"{v}.png"))
    for v, d in emotion_original.items():
        barplot_counts(d, v, os.path.join(save_dir, f"original_{v}.png"))
    for v, d in emotion_baseline.items():
        barplot_counts(d, v, os.path.join(save_dir, f"baseline_{v}.png"))
    for v, d in emotion_proposed.items():
        barplot_counts(d, v, os.path.join(save_dir, f"proposed_{v}.png"))
    for v, d in emotion_prompt.items():
        barplot_counts(d, v, os.path.join(save_dir, f"prompt_{v}.png"))

    barplot_counts_all(emotion_original, os.path.join(save_dir, f"emotion_original.png"))
    barplot_counts_all(emotion_baseline, os.path.join(save_dir, f"emotion_baseline.png"))
    barplot_counts_all(emotion_proposed, os.path.join(save_dir, f"emotion_proposed.png"))
    barplot_counts_all(emotion_prompt, os.path.join(save_dir, f"emotion_prompt.png"))

    heatmap_counts_all(emotion_original, os.path.join(save_dir, f"heatmap_emotion_original.png"), "m")
    heatmap_counts_all(emotion_baseline, os.path.join(save_dir, f"heatmap_emotion_baseline.png"), "m")
    heatmap_counts_all(emotion_proposed, os.path.join(save_dir, f"heatmap_emotion_proposed.png"), "m")
    heatmap_counts_all(emotion_prompt, os.path.join(save_dir, f"heatmap_emotion_prompt.png"), "m")
    
    barplot_mean_ratings(sim_mean, os.path.join(save_dir, f"sim_mean.png"))
    barplot_mean_ratings(mos_original, os.path.join(save_dir, f"mos_original.png"))
    barplot_mean_ratings(mos_baseline, os.path.join(save_dir, f"mos_baseline.png"))
    barplot_mean_ratings(mos_proposed, os.path.join(save_dir, f"mos_proposed.png"))
    barplot_mean_ratings(mos_prompt, os.path.join(save_dir, f"mos_prompt.png"))
    barplot_mean_ratings(mos, os.path.join(save_dir, f"mos.png"))

    scatterplot_va(va_original, os.path.join(save_dir, f"va_original.png"))
    scatterplot_va(va_baseline, os.path.join(save_dir, f"va_baseline.png"))
    scatterplot_va(va_proposed, os.path.join(save_dir, f"va_proposed.png"))
    scatterplot_va(va_prompt, os.path.join(save_dir, f"va_prompt.png"))
