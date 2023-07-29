import pandas as pd
import numpy as np

EMOTIONS = ["anger", "joy", "neutral", "sadness", "surprise"]

def index_to_emotion(i):
    id_emotion = {1: "anger", 2: "joy", 3: "neutral", 4: "sadness", 5: "surprise"}
    return id_emotion[i]

def read_data(path_to_data):
    return pd.read_csv(path_to_data, encoding="utf-8", delimiter=";")

def sociodemographics(data):
    d = {}
    d["age"] = dict(data["SD03"].value_counts().sort_index())
    d["gender"] = dict(data["SD01"].value_counts().sort_index())
    d["english_skills"] = dict(data["SD20"].value_counts().sort_index())
    d["experience"] = dict(data["SD19"].value_counts().sort_index())
    return d

def preference(data):
    d = {}
    # A baseline, B proposed
    d["pref_anger_0"] = dict(data["CP01"].value_counts().sort_index())
    d["pref_anger_1"] = dict(data["CP02"].value_counts().sort_index())
    d["pref_joy_0"] = dict(data["CP03"].value_counts().sort_index())
    d["pref_joy_1"] = dict(data["CP04"].value_counts().sort_index())
    d["pref_neutral_0"] = dict(data["CP05"].value_counts().sort_index())
    # A proposed, B baseline, so keys are switched such that the order is always the same as above
    d_tmp = dict(data["CP06"].value_counts().sort_index())
    d["pref_neutral_1"] = {1.0: d_tmp.get(2.0), 2.0: d_tmp.get(1.0), 3.0: d_tmp.get(3.0)}
    d_tmp = dict(data["CP07"].value_counts().sort_index())
    d["pref_sadness_0"] = {1.0: d_tmp.get(2.0), 2.0: d_tmp.get(1.0), 3.0: d_tmp.get(3.0)}
    d_tmp = dict(data["CP08"].value_counts().sort_index())
    d["pref_sadness_1"] = {1.0: d_tmp.get(2.0), 2.0: d_tmp.get(1.0), 3.0: d_tmp.get(3.0)}
    d_tmp = dict(data["CP09"].value_counts().sort_index())
    d["pref_surprise_0"] = {1.0: d_tmp.get(2.0), 2.0: d_tmp.get(1.0), 3.0: d_tmp.get(3.0)}
    d_tmp = dict(data["CP10"].value_counts().sort_index())
    d["pref_surprise_1"] = {1.0: d_tmp.get(2.0), 2.0: d_tmp.get(1.0), 3.0: d_tmp.get(3.0)}
    return d

def similarity(data):
    d = {}
    d["sim_anger_0"] = dict(data["CS01_01"].value_counts().sort_index())
    d["sim_anger_1"] = dict(data["CS02_01"].value_counts().sort_index())
    d["sim_joy_0"] = dict(data["CS03_01"].value_counts().sort_index())
    d["sim_joy_1"] = dict(data["CS04_01"].value_counts().sort_index())
    d["sim_neutral_0"] = dict(data["CS05_01"].value_counts().sort_index())
    d["sim_neutral_1"] = dict(data["CS06_01"].value_counts().sort_index())
    d["sim_sadness_0"] = dict(data["CS07_01"].value_counts().sort_index())
    d["sim_sadness_1"] = dict(data["CS08_01"].value_counts().sort_index())
    d["sim_surprise_0"] = dict(data["CS09_01"].value_counts().sort_index())
    d["sim_surprise_1"] = dict(data["CS10_01"].value_counts().sort_index())
    return d

def mean_opinion_score(data, version):
    d = {}
    d["mos_anger_0"] = dict(data[f"M{version}01"].value_counts().sort_index())
    d["mos_anger_1"] = dict(data[f"M{version}02"].value_counts().sort_index())
    d["mos_joy_0"] = dict(data[f"M{version}03"].value_counts().sort_index())
    d["mos_joy_1"] = dict(data[f"M{version}04"].value_counts().sort_index())
    d["mos_neutral_0"] = dict(data[f"M{version}05"].value_counts().sort_index())
    d["mos_neutral_1"] = dict(data[f"M{version}06"].value_counts().sort_index())
    d["mos_sadness_0"] = dict(data[f"M{version}07"].value_counts().sort_index())
    d["mos_sadness_1"] = dict(data[f"M{version}08"].value_counts().sort_index())
    d["mos_surprise_0"] = dict(data[f"M{version}09"].value_counts().sort_index())
    d["mos_surprise_1"] = dict(data[f"M{version}10"].value_counts().sort_index())
    return d

def emotion(data, version):
    d = {}
    d["emotion_anger_0"] = {}
    d["emotion_anger_1"] = {}
    d["emotion_joy_0"] = {}
    d["emotion_joy_1"] = {}
    d["emotion_neutral_0"] = {}
    d["emotion_neutral_1"] = {}
    d["emotion_sadness_0"] = {}
    d["emotion_sadness_1"] = {}
    d["emotion_surprise_0"] = {}
    d["emotion_surprise_1"] = {}
    variable_count = 1
    for emo in EMOTIONS:
        for j in range(2):
            for k, emo_count in enumerate(EMOTIONS):
                try:
                    variable = f"0{variable_count}" if variable_count < 10 else variable_count
                    d[f"emotion_{emo}_{j}"][emo_count] = dict(data[f"E{version}{variable}_0{k+1}"].value_counts().sort_index())[2]
                except KeyError:
                    d[f"emotion_{emo}_{j}"][emo_count] = 0
            variable_count += 1
    return d

def valence_arousal(data, version):
    d = {"valence": {}, "arousal": {}}
    d["valence"]["anger_0"] = data[f"V{version}01_01"].mean()
    d["valence"]["anger_1"] = data[f"V{version}02_01"].mean()
    d["valence"]["joy_0"] = data[f"V{version}03_01"].mean()
    d["valence"]["joy_1"] = data[f"V{version}04_01"].mean()
    d["valence"]["neutral_0"] = data[f"V{version}05_01"].mean()
    d["valence"]["neutral_1"] = data[f"V{version}06_01"].mean()
    d["valence"]["sadness_0"] = data[f"V{version}07_01"].mean()
    d["valence"]["sadness_1"] = data[f"V{version}08_01"].mean()
    d["valence"]["surprise_0"] = data[f"V{version}09_01"].mean()
    d["valence"]["surprise_1"] = data[f"V{version}10_01"].mean()

    d["arousal"]["anger_0"] = data[f"V{version}01_02"].mean()
    d["arousal"]["anger_1"] = data[f"V{version}02_02"].mean()
    d["arousal"]["joy_0"] = data[f"V{version}03_02"].mean()
    d["arousal"]["joy_1"] = data[f"V{version}04_02"].mean()
    d["arousal"]["neutral_0"] = data[f"V{version}05_02"].mean()
    d["arousal"]["neutral_1"] = data[f"V{version}06_02"].mean()
    d["arousal"]["sadness_0"] = data[f"V{version}07_02"].mean()
    d["arousal"]["sadness_1"] = data[f"V{version}08_02"].mean()
    d["arousal"]["surprise_0"] = data[f"V{version}09_02"].mean()
    d["arousal"]["surprise_1"] = data[f"V{version}10_02"].mean()
    return d

def get_mean_rating_nested(d: dict):
    d_mean = {}
    for k, v in d.items():
        total_sum = 0
        count = 0
        for rating, count_value in v.items():
            total_sum += rating * count_value
            count += count_value
        mean_rating = total_sum / count
        d_mean[k] = mean_rating
    return d_mean

def split_female_male(combined):
    female = {}
    male = {}
    for sent_id, d in combined.items():
        try:
            if sent_id.split("_")[1] + "_" + sent_id.split("_")[2] in emo_sent_speaker("f"):
                female[sent_id.split("_")[1]] = d
            if sent_id.split("_")[1] + "_" + sent_id.split("_")[2] in emo_sent_speaker("m"):
                male[sent_id.split("_")[1]] = d
        except IndexError:
            if sent_id in emo_sent_speaker("f"):
                female[sent_id.split("_")[0]] = d
            if sent_id in emo_sent_speaker("m"):
                male[sent_id.split("_")[0]] = d
    return female, male

def make_emotion_prompts(emotion_d, speaker):
    emo_prompt_match = emo_prompt_speaker(speaker)
    emotion_prompt_d = {}
    for emo, d in emotion_d.items():
        emotion_prompt_d[emo_prompt_match[emo]] = d
    return emotion_prompt_d

def emo_sent_speaker(speaker):
    if speaker ==  "f":
        return ["anger_0", "joy_0", "neutral_0", "sadness_0", "surprise_1"]
    else:
        return ["anger_1", "joy_1", "neutral_1", "sadness_1", "surprise_0"]
    
def emo_prompt_speaker(speaker):
    if speaker == "f":
        return {"anger"   : "neutral",
                "joy"     : "surprise",
                "neutral" : "joy",
                "sadness" : "anger",
                "surprise": "sadness"}
    else:
        return {"anger"   : "surprise",
                "joy"     : "sadness",
                "neutral" : "anger",
                "sadness" : "joy",
                "surprise": "neutral"}
    
def remove_outliers(data):
    ratings = {emotion: list(rating for rating, count in ratings_counts.items() for _ in range(count)) for emotion, ratings_counts in data.items()}
    cleaned_data = {}
    for emotion, ratings_list in ratings.items():
        sorted_data = sorted(ratings_list)
        q1, q3 = np.percentile(sorted_data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        cleaned_data[emotion] = [x for x in sorted_data if lower_bound <= x <= upper_bound]
    counts = {}
    for emotion, ratings in cleaned_data.items():
        rating_counts = {}
        for rating in ratings:
            if rating in rating_counts:
                rating_counts[rating] += 1
            else:
                rating_counts[rating] = 1
        counts[emotion] = rating_counts
    return counts

def combine_dicts(d1, d2):
    combined_dict = {}
    for emotion in d1:
        combined_dict[emotion] = {}
        all_keys = sorted(set(d1[emotion].keys()) | set(d2[emotion].keys()))
        for rating in all_keys:
            combined_dict[emotion][rating] = d1[emotion].get(rating, 0) + d2[emotion].get(rating, 0)
    return combined_dict

def collapse_subdicts(d):
    collapsed_dict = {}
    for emotion, sub_dict in d.items():
        for scenario, count in sub_dict.items():
            if scenario not in collapsed_dict:
                collapsed_dict[scenario] = 0
            collapsed_dict[scenario] += count
    return collapsed_dict
