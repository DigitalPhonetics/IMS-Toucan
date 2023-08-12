import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

EMOTIONS = ["anger", "joy", "neutral", "sadness", "surprise"]
COLORS = ['red', 'green', 'blue', 'gray', 'orange']

def barplot_counts(d: dict, v, save_dir):
    labels = get_variable_labels(v)
    if labels is None:
        labels = {k:k for k in list(d.keys())}
    values = [d[label] for label in labels if label in d]

    plt.bar(range(len(values)), values, align='center')
    plt.xticks(range(len(values)), [labels[label] for label in labels if label in d])
    plt.xlabel(v)
    plt.ylabel('counts')
    plt.savefig(save_dir)
    plt.close()

def pie_chart_counts(d: dict, v, save_dir):
    labels = get_variable_labels(v)
    if labels is None:
        labels = {k:k for k in list(d.keys())}
    relevant_labels = set(labels.keys()) & set(d.keys())
    #values = [d[label] for label in labels if label in d]
    values = [d[label] for label in relevant_labels]
    total = sum(values)
    labels = [labels[label] for label in relevant_labels]
    percentages = [(value / total) * 100 for value in values]

    colors = plt.cm.Set3(range(len(labels)))

    plt.figure(figsize=(8, 8))
    plt.pie(percentages, labels=labels, autopct=lambda p: '{:.1f}%'.format(p) if p >= 2 else '', colors=colors, startangle=90, textprops={'fontsize': 16})
    plt.axis('equal')
    plt.rcParams['font.size'] = 12
    plt.savefig(save_dir)
    plt.close()

def barplot_pref(d, save_dir):
    # Define the emotions and their corresponding colors
    emotions = EMOTIONS
    colors = COLORS

    # Define the x-axis tick labels
    x_ticks = ['Baseline', 'Proposed', 'No Preference']

    # Initialize the plot
    fig, ax = plt.subplots()

    # Set the x-axis tick locations and labels
    ax.set_xticks(np.arange(len(x_ticks)))
    ax.set_xticklabels(x_ticks)

    # Calculate the width of each bar
    bar_width = 0.15

    # Calculate the offset for each emotion's bars
    offsets = np.linspace(-2 * bar_width, 2 * bar_width, len(emotions))

    # Calculate the total count for each emotion
    total_counts = [sum(d[emotion].values()) for emotion in emotions]

    # Iterate over the emotions and plot the bars
    for i, emotion in enumerate(emotions):
        counts = [d[emotion].get(1.0, 0), d[emotion].get(2.0, 0), d[emotion].get(3.0, 0)]
        percentages = [count / total_counts[i] * 100 for count in counts]
        positions = np.arange(len(percentages)) + offsets[i]
        ax.bar(positions, percentages, width=bar_width, color=colors[i], label=emotion)

    # Set the legend
    ax.legend()

    # Set the plot title and axis labels
    ax.set_ylabel('Percentage (%)')

    # Adjust the x-axis limits and labels
    ax.set_xlim(-2 * bar_width - bar_width, len(x_ticks) - 1 + 2 * bar_width + bar_width)
    ax.set_xticks(np.arange(len(x_ticks)))

    # Save the plot
    plt.savefig(save_dir)
    plt.close()

def barplot_pref2(d, save_dir):
    # Define the emotions and their corresponding colors
    emotions = EMOTIONS
    colors = COLORS
    emotions = emotions[::-1]
    colors = colors[::-1]
    patterns = ['///', '\\\\\\', 'xxx']

    # Define the y-axis tick labels
    y_ticks = emotions

    # Calculate the height of each bar
    bar_height = 0.5

    # Initialize the plot
    fig, ax = plt.subplots()

    # Set the y-axis tick locations and labels
    ax.set_yticks(np.arange(len(y_ticks)))
    ax.set_yticklabels(y_ticks)

    # Calculate the width of each bar segment
    total_counts = [d[emotion].get(1.0, 0) + d[emotion].get(2.0, 0) + d[emotion].get(3.0, 0) for emotion in emotions]
    baseline_counts = [d[emotion].get(1.0, 0) for emotion in emotions]
    proposed_counts = [d[emotion].get(2.0, 0) for emotion in emotions]
    no_pref_counts = [d[emotion].get(3.0, 0) for emotion in emotions]
    percentages = [count / sum(total_counts) * 100 for count in total_counts]
    baseline_percentages = [count / total * 100 if total != 0 else 0 for count, total in zip(baseline_counts, total_counts)]
    proposed_percentages = [count / total * 100 if total != 0 else 0 for count, total in zip(proposed_counts, total_counts)]
    no_pref_percentages = [count / total * 100 if total != 0 else 0 for count, total in zip(no_pref_counts, total_counts)]
    total_width = np.max(percentages)

    # Plot the bars with different shadings and colors
    for i in range(len(emotions)):
        # Plot only the first bar of each emotion with the corresponding label
        ax.barh(y_ticks[i], baseline_percentages[i], height=bar_height, color=colors[i], edgecolor='black', hatch=patterns[0], label='Baseline' if i == 0 else '')
        ax.barh(y_ticks[i], no_pref_percentages[i], height=bar_height, color=colors[i], edgecolor='black', hatch=patterns[2], left=baseline_percentages[i], label='No Preference' if i == 0 else '')
        ax.barh(y_ticks[i], proposed_percentages[i], height=bar_height, color=colors[i], edgecolor='black', hatch=patterns[1], left=np.add(baseline_percentages[i], no_pref_percentages[i]), label='Proposed' if i == 0 else '')
        ax.text(total_width + 10, i, emotions[i], color=colors[i], verticalalignment='center')

    # Set the legend outside the plot
    legend_labels = [plt.Rectangle((0, 0), 0, 0, edgecolor='black', hatch=patterns[i], facecolor='white') for i in range(len(patterns))]
    legend_names = ['Baseline', 'Proposed', 'No Preference']
    ax.legend(legend_labels, legend_names, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, edgecolor='black', facecolor='none')

    # Set the plot title and axis labels
    ax.set_xlabel('Percentage (%)')

    # Adjust the y-axis limits and labels
    ax.set_ylim(-0.5, len(y_ticks) - 0.5)
    ax.set_yticks(np.arange(len(y_ticks)))
    for tick_label, color in zip(ax.get_yticklabels(), colors):
        tick_label.set_color(color)

    # Save the plot
    plt.savefig(save_dir)
    plt.close()

def barplot_pref3(d, save_dir):
    # Define the emotions and their corresponding colors
    emotions = EMOTIONS
    colors = ['black', 'gray', 'lightgray']  # Colors for Original, Baseline, and Proposed bars
    emotion_colors = COLORS

    # Define the x-axis tick labels (emotions)
    x_ticks = emotions

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(8,6))

    # Set the x-axis tick locations and labels
    ax.set_xticks(np.arange(len(x_ticks)))
    labels = ax.set_xticklabels(x_ticks)
    for label, color in zip(labels, emotion_colors):
        label.set_color(color)

    # Calculate the width of each bar group
    bar_width = 0.2

    # Calculate the offset for each scenario's bars within each group
    offsets = np.linspace(-bar_width, bar_width, 3)

    scenarios = ['Baseline', 'Proposed', 'No Preference']
    # Iterate over the scenarios and plot the bars for each emotion
    for i, scenario in enumerate([1.0, 2.0, 3.0]):
        percentages = [d[emotion].get(scenario, 0) / sum(d[emotion].values()) * 100 for emotion in emotions]
        positions = np.arange(len(percentages)) + offsets[i]
        ax.bar(positions, percentages, width=bar_width, color=colors[i], label=scenarios[i])

    # Set the legend
    ax.legend()

    # Set the plot title and axis labels
    ax.set_ylabel('Percentage (%)', fontsize=16)

    # Adjust the x-axis limits and labels
    ax.set_xlim(-bar_width * 2, len(x_ticks) - 1 + bar_width * 2)
    ax.set_xticks(np.arange(len(x_ticks)))
    ax.set_ylim(0, 100)

    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save the plot
    plt.savefig(save_dir)
    plt.close()

def barplot_pref_total(d, save_dir):

    # Define the x-axis tick labels
    x_ticks = ['Baseline', 'Proposed', 'No Preference']
    colors = ['black', 'gray', 'lightgray'] 

    # Initialize the plot
    fig, ax = plt.subplots()

    # Set the x-axis tick locations and labels
    ax.set_xticks(np.arange(len(x_ticks)))
    ax.set_xticklabels(x_ticks)

    # Calculate the width of each bar
    bar_width = 0.5

    # Calculate the offset for each emotion's bars
    offsets = np.linspace(-2 * bar_width, 2 * bar_width, len(x_ticks))

    # Calculate the total count for each emotion
    total_counts = sum(list(d.values()))

    # Iterate over the emotions and plot the bars
    for i, scenario in enumerate([1.0, 2.0, 3.0]):
        counts = [d.get(scenario, 0)]
        percentages = [count / total_counts * 100 for count in counts]
        positions = i
        ax.bar(positions, percentages, width=bar_width, color=colors[i], align='center')
        plt.text(i, round(percentages[0], 2), str(round(percentages[0], 2)), ha='center', va='bottom', fontsize=14)

    # Set the plot title and axis labels
    ax.set_ylabel('Percentage (%)', fontsize=16)

    # Adjust the x-axis limits and labels
    #ax.set_xlim(-2 * bar_width - bar_width, len(x_ticks) - 1 + 2 * bar_width + bar_width)
    ax.set_xticks(np.arange(len(x_ticks)))
    ax.set_ylim(0, 60)

    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save the plot
    plt.savefig(save_dir)
    plt.close()

def barplot_sim(data, save_dir):
    emotions = EMOTIONS
    ratings = [data[emotion] for emotion in emotions]
    colors = COLORS

    fig, ax = plt.subplots(figsize=(8,6))

    x_ticks = emotions

    ax.set_xticks(np.arange(len(x_ticks)))
    labels = ax.set_xticklabels(x_ticks, fontsize=16)
    for label, color in zip(labels, colors):
        label.set_color(color)

    bar_width = 0.5

    positions = np.arange(len(ratings))

    bars = ax.bar(positions, ratings, width=bar_width, color=colors)

    for bar, rating in zip(bars, ratings):
        ax.annotate(f'{rating:.2f}',  # Format the rating to one decimal place
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),  # 3 points vertical offset from the bar
                    textcoords='offset points',
                    ha='center', va='bottom', fontsize=14)

    ax.set_ylabel('Mean Similatrity Score', fontsize=16)

    ax.set_yticks(np.arange(6))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(save_dir)
    plt.close()

def barplot_sim_total(data, save_dir):
    speakers = list(data.keys())
    ratings = list(data.values())

    fig, ax = plt.subplots()

    x_ticks = speakers

    ax.set_xticks(np.arange(len(x_ticks)))
    ax.set_xticklabels(x_ticks, fontsize=16)

    bar_width = 0.5

    positions = np.arange(len(ratings))

    bars = ax.bar(positions, ratings, width=bar_width, color='gray')

    for bar, rating in zip(bars, ratings):
        ax.annotate(f'{rating:.2f}',  # Format the rating to one decimal place
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),  # 3 points vertical offset from the bar
                    textcoords='offset points',
                    ha='center', va='bottom', fontsize=14)

    ax.set_ylabel('Mean Similatrity Score', fontsize=16)

    ax.set_yticks(np.arange(6))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(save_dir)
    plt.close()

def boxplot_rating(data, save_dir):
    emotions = EMOTIONS
    ratings = {emotion: list(rating for rating, count in data[emotion].items() for _ in range(count)) for emotion in emotions}
    colors = COLORS

    plt.figure(figsize=(10, 6))
    box_plot = plt.boxplot(ratings.values(), patch_artist=True, widths=0.7)
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    for median in box_plot['medians']:
        median.set(color='black', linestyle='-', linewidth=3)
    plt.xticks(range(1, len(emotions) + 1), emotions, fontsize=16)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0.9, 5.1)
    ax.tick_params(axis='y', labelsize=16)
    ax.yaxis.set_major_locator(MultipleLocator(base=1))
    for i, t in enumerate(ax.xaxis.get_ticklabels()):
        t.set_color(colors[i])
    plt.savefig(save_dir)
    plt.close()

def barplot_mos(l: list, save_dir):
    values = l
    labels = ["Ground Truth", "Baseline", "Proposed"]

    plt.bar(range(len(values)), values, align='center', color="gray")
    plt.xticks(range(len(values)), labels, fontsize=16)
    plt.ylabel('MOS', fontsize=16)
    ax = plt.gca()
    ax.set_yticks(np.arange(6))
    ax.set_ylim(0.9, 5.1)
    ax.tick_params(axis='y', labelsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i, v in enumerate(values):
        plt.text(i, round(v, 2), str(round(v, 2)), ha='center', va='bottom', fontsize=14)
    plt.savefig(save_dir)
    plt.close()

def barplot_emotion(d: dict, save_dir):
    emotions = EMOTIONS
    colors = COLORS
    bar_width = 0.5

    d = {key: d[key] for key in sorted(d, key=lambda x: emotions.index(x))}

    fig, ax = plt.subplots()

    subdicts = d.items()
    num_subdicts = len(subdicts)
    num_labels = len(list(subdicts)[0][1])  # Assuming all sub-dicts have the same labels

    total_width = bar_width * num_subdicts
    group_width = bar_width / num_subdicts
    x = np.arange(num_labels) - (total_width / 2) + (group_width / 2)

    color_id = 0
    for i, (v, subdict) in enumerate(subdicts):
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
    ax.legend(emotions, loc='upper right', bbox_to_anchor=(0.9, 1))
    ax.set_ylabel('Counts')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(save_dir)
    plt.close()

def heatmap_emotion(d: dict, save_dir):
    emotions = EMOTIONS

    # Create a numpy array to store the counts
    counts = np.array([[d[emotion].get(label, 0) for emotion in emotions] for label in emotions])
    # normalize counts for each emotion category
    normalized_counts = np.around(counts / counts.sum(axis=0, keepdims=True), decimals=2)

    # Set up the figure and heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(normalized_counts, cmap='viridis', vmin=0, vmax=1)

    # Show counts as text in each cell
    for i in range(len(emotions)):
        for j in range(len(emotions)):
            text = ax.text(i, j, normalized_counts[j, i], ha='center', va='center', color='black', fontsize=16)

    # Set the axis labels and title
    ax.set_xticks(np.arange(len(emotions)))
    ax.set_yticks(np.arange(len(emotions)))
    ax.set_xticklabels(emotions, fontsize=16)
    ax.set_yticklabels(emotions, fontsize=16)

    # Rotate the tick labels for better readability (optional)
    plt.xticks(rotation=45, ha='right', fontsize=16)

    # Create a colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('Relative Frequency', fontsize=16)

    # Save the heatmap
    plt.savefig(save_dir)
    plt.close()

def scatterplot_va(v: dict, a: dict, save_dir: str):
    # Initialize a color map for differentiating emotions
    colors = COLORS

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Iterate over emotions and variations
    color_id = 0
    for emo in EMOTIONS:
        valence = v[emo]
        arousal = a[emo]

        # Plot a single point with a unique color
        ax.scatter(valence, arousal, color=colors[color_id], label=emo)
        color_id += 1

    # Set plot title and labels
    #ax.set_xlabel("Valence")
    #ax.set_ylabel("Arousal")

    # Set axis limits and center the plot at (3, 3)
    ax.set_xlim(0.9, 5.1)
    ax.set_ylim(0.9, 5.1)
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

def boxplot_objective(data, save_dir):
    data = dict(sorted(data.items()))
    speakers = list(data.keys())
    ratings = list(data.values())

    plt.figure(figsize=(10, 6))
    box_plot = plt.boxplot(ratings, patch_artist=True, widths=0.7)
    for patch in box_plot['boxes']:
        patch.set_facecolor('white')
    for median in box_plot['medians']:
        median.set(color='black', linestyle='-', linewidth=3)
    plt.xticks(range(1, len(speakers) + 1), speakers, fontsize=16)
    plt.xlabel('Speaker', fontsize=16)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=16)
    ax.yaxis.set_major_locator(MultipleLocator(base=0.01))
    plt.savefig(save_dir)
    plt.close()

def boxplot_objective2(data, save_dir):
    data = dict(sorted(data.items()))
    speakers = list(data.keys())
    ratings = list(data.values())

    plt.figure(figsize=(10, 6))
    box_plot = plt.boxplot(ratings, patch_artist=True, widths=0.7)
    for patch in box_plot['boxes']:
        patch.set_facecolor('white')
    for median in box_plot['medians']:
        median.set(color='black', linestyle='-', linewidth=3)
    plt.xticks(range(1, len(speakers) + 1), speakers, fontsize=16)
    plt.xlabel('Speaker', fontsize=16)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=16)
    ax.yaxis.set_major_locator(MultipleLocator(base=0.1))
    plt.savefig(save_dir)
    plt.close()

def barplot_speaker_similarity(data, save_dir):
    labels = ['Baseline', 'Proposed']

    plt.bar(range(len(data)), data, align='center', color="gray")
    plt.xticks(range(len(data)), labels, fontsize=16)
    plt.ylabel('Cosine Similarity', fontsize=16)
    ax = plt.gca()
    ax.tick_params(axis='y', labelsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i, v in enumerate(data):
        plt.text(i, round(v, 4), str(round(v, 4)), ha='center', va='bottom', fontsize=14)
    plt.savefig(save_dir)
    plt.close()

def barplot_wer(data, save_dir):
    labels = ['Ground Thruth', 'Baseline', 'Proposed']

    plt.bar(range(len(data)), data, align='center', color="gray")
    plt.xticks(range(len(data)), labels, fontsize=16)
    plt.ylabel('Word Error Rate', fontsize=16)
    ax = plt.gca()
    ax.tick_params(axis='y', labelsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i, v in enumerate(data):
        plt.text(i, round(v, 3), str(round(v, 3)), ha='center', va='bottom', fontsize=14)
    plt.subplots_adjust(left=0.15)
    plt.savefig(save_dir)
    plt.close()

def barplot_emotion_recognition(data, save_dir):
    labels = ['Ground Truth', 'Baseline', 'Proposed Same', 'Proposed Other']

    # Set up the figure with a larger width to accommodate tick labels
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use a horizontal bar plot with reversed data
    bars = ax.barh(range(len(data)), data[::-1], color="gray")  # Reversed data

    # Set the tick positions and labels
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(labels[::-1], fontsize=16)  # Reversed labels

    # Set the x-axis label
    ax.set_xlabel('Accuracy', fontsize=16)

    # Set the font size for y-axis tick labels
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=16)

    # Remove the right and top spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add accuracy values as text beside each bar
    for i, v in enumerate(data[::-1]):  # Reversed data
        ax.text(v, i, str(round(v, 2)), ha='left', va='center', fontsize=14)  # Reversed data

    # Adjust the plot layout to prevent the labels from being cut off
    plt.subplots_adjust(left=0.2, right=0.95)

    # Save the bar plot
    plt.savefig(save_dir)
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
