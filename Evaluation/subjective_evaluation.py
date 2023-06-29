import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
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

def barplot_counts_all(d: dict, save_dir):
    colors = ['red', 'green', 'blue', 'gray', 'orange']
    bar_width = 0.5

    plt.clf()
    fig, ax = plt.subplots()

    subdicts = d.items()
    num_subdicts = len(subdicts)
    num_labels = len(list(subdicts)[0][1])  # Assuming all sub-dicts have the same labels

    total_width = bar_width * num_subdicts
    group_width = bar_width / num_subdicts
    x = np.arange(num_labels) - (total_width / 2) + (group_width / 2)

    color_id = 0
    variables = []
    for i, (v, subdict) in enumerate(subdicts):
        if v.split("emotion_")[1] in emo_sent_speaker("f"):
            variables.append(v.split("_")[1])
            values = [subdict[label] for label in subdict]
            offset = i * group_width
            color = colors[color_id]
            ax.bar(x + offset, values, group_width, align='center', color=color)
            color_id += 1

    ax.set_xticks(x + 0.2)
    ax.set_xticklabels(list(list(subdicts)[0][1].keys()))
    for i, t in enumerate(ax.xaxis.get_ticklabels()):
        t.set_color(colors[i])
    ax.tick_params(axis='x', length=0)
    ax.legend(variables, loc='upper right', bbox_to_anchor=(0.9, 1))
    ax.set_ylabel('Counts')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(save_dir)
    plt.close()

import numpy as np
import matplotlib.pyplot as plt

def heatmap_counts_all(d: dict, save_dir, speaker):
    colors = 'coolwarm'

    plt.clf()
    fig, ax = plt.subplots()

    subdicts = d.items()
    num_subdicts = 5
    num_labels = 5  # Assuming all sub-dicts have the same labels

    data = np.zeros((num_labels, num_subdicts))
    xticklabels = []

    i = 0
    for v, subdict in subdicts:
        if v.split("emotion_")[1] in emo_sent_speaker(speaker):
            xticklabels.append(v.split("_")[1])
            values = [subdict[label] for label in subdict]
            data[:, i] = values
            i += 1

    im = ax.imshow(data, cmap=colors)

    ax.set_xticks(np.arange(num_subdicts))
    ax.set_yticks(np.arange(num_labels))
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(list(list(subdicts)[0][1].keys()))

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(num_labels):
        for j in range(num_subdicts):
            text = ax.text(j, i, int(data[i, j]), ha="center", va="center", color="w")

    ax.set_xlabel('Variables')
    ax.set_ylabel('Labels')
    ax.set_title('Heatmap Counts')

    plt.savefig(save_dir)
    plt.close()

def barplot_mean_ratings(d: dict, save_dir):
    values = list(d.values())

    plt.clf()
    plt.bar(range(len(values)), values, align='center')
    plt.xticks(range(len(values)), list(d.keys()))
    plt.xlabel('')
    plt.ylabel('rating')
    plt.savefig(save_dir)

def scatterplot_va(d: dict, save_dir: str):
    # Initialize a color map for differentiating emotions
    color_map = plt.cm.get_cmap("tab10")

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Iterate over emotions and variations
    color_id = 0
    for emo in EMOTIONS:
        for i in range(2):
            valence = d["valence"][f"{emo}_{i}"]
            arousal = d["arousal"][f"{emo}_{i}"]

            # Plot a single point with a unique color
            ax.scatter(valence, arousal, color=color_map(color_id), label=f"{emo}_{i}")
            color_id += 1

    # Set plot title and labels
    #ax.set_xlabel("Valence")
    #ax.set_ylabel("Arousal")

    # Set axis limits and center the plot at (3, 3)
    ax.set_xlim(1, 5)
    ax.set_ylim(1, 5)
    ax.set_xticks([1, 5])
    ax.set_yticks([1, 5])
    ax.set_xticklabels(['negative', 'positive'])
    ax.set_yticklabels(['calm', 'excited'])
    ax.set_aspect('equal')
    ax.spines['left'].set_position(('data', 3))
    ax.spines['bottom'].set_position(('data', 3))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Add a legend in the top-left corner
    legend = ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), bbox_transform=plt.gcf().transFigure)

    # Adjust the plot layout to accommodate the legend
    plt.subplots_adjust(top=0.95, right=0.95)

    # Save the figure
    plt.savefig(save_dir, bbox_inches='tight')
    plt.close()

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

def emo_sent_speaker(speaker):
    if speaker ==  "f":
        return ["anger_0", "joy_0", "neutral_0", "sadness_0", "surprise_1"]
    else:
        return ["anger_1", "joy_1", "neutral_1", "sadness_1", "surprise_0"]
