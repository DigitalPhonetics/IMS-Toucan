import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean

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
    d["pref_anger_0"] = dict(data["CP01"].value_counts().sort_index())
    d["pref_anger_1"] = dict(data["CP02"].value_counts().sort_index())
    d["pref_joy_0"] = dict(data["CP03"].value_counts().sort_index())
    d["pref_joy_1"] = dict(data["CP04"].value_counts().sort_index())
    d["pref_neutral_0"] = dict(data["CP05"].value_counts().sort_index())
    d["pref_neutral_1"] = dict(data["CP06"].value_counts().sort_index())
    d["pref_sadness_0"] = dict(data["CP07"].value_counts().sort_index())
    d["pref_sadness_1"] = dict(data["CP08"].value_counts().sort_index())
    d["pref_surprise_0"] = dict(data["CP09"].value_counts().sort_index())
    d["pref_surprise_1"] = dict(data["CP10"].value_counts().sort_index())
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
    d["mos_anger_0"] = data[f"M{version}01"].mean()
    d["mos_anger_1"] = data[f"M{version}02"].mean()
    d["mos_joy_0"] = data[f"M{version}03"].mean()
    d["mos_joy_1"] = data[f"M{version}04"].mean()
    d["mos_neutral_0"] = data[f"M{version}05"].mean()
    d["mos_neutral_1"] = data[f"M{version}06"].mean()
    d["mos_sadness_0"] = data[f"M{version}07"].mean()
    d["mos_sadness_1"] = data[f"M{version}08"].mean()
    d["mos_surprise_0"] = data[f"M{version}09"].mean()
    d["mos_surprise_1"] = data[f"M{version}10"].mean()
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

def barplot_counts(d: dict, v, save_dir):
    labels = get_variable_labels(v)
    if labels is None:
        labels = {k:k for k in list(d.keys())}
    values = [d[label] for label in labels if label in d]

    plt.clf()
    plt.bar(range(len(values)), values, align='center')
    plt.xticks(range(len(values)), [labels[label] for label in labels if label in d])
    plt.xlabel(v)
    plt.ylabel('counts')
    plt.savefig(save_dir)

def barplot_mean_ratings(d: dict, save_dir):
    values = list(d.values())

    plt.clf()
    plt.bar(range(len(values)), values, align='center')
    plt.xticks(range(len(values)), list(d.keys()))
    plt.xlabel('')
    plt.ylabel('rating')
    plt.savefig(save_dir)

def get_variable_labels(v):
    labels = None

    if v == "age":
        labels = {1: "<20",
                  2: "20-29",
                  3: "30-39",
                  4: "40-49",
                  5: "50-59",
                  6: "60-69",
                  7: "70-79",
                  8: ">80",
                  -9: "--"}
    if v == "gender":
        labels = {1: "female",
                  2: "male",
                  3: "divers",
                  -9: "--"}
    if v == "english_skills":
        labels = {1: "none",
                  2: "beginner",
                  3: "intermediate",
                  4: "advanced",
                  5: "fluent",
                  6: "native",
                  -9: "--"}
    if v == "experience":
        labels = {1: "daily",
                  2: "regularly",
                  3: "rarely",
                  4: "never",
                  -9: "--"}
    if v.startswith("pref_"):
        labels = {1: "A",
                  2: "B",
                  3: "no preference"}
    if v.startswith("sim_"):
        labels = {1: "1",
                  2: "2",
                  3: "3",
                  4: "4",
                  5: "5"}
        
    return labels
