import json

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ArticulatoryCombinedTextFrontend import ArticulatoryCombinedTextFrontend


def plot_embeddings(reduced_data, phoneme_list, title):
    consonants = ['w', 'b', 'ɡ', 'n', 'ʒ', 'ʃ', 'd', 'l', 'θ', 'ŋ', 'f', 'ɾ', 's', 'm', 't', 'h', 'z', 'p', 'ʔ', 'v', 'ɹ', 'j', 'ð', 'k']
    vowels = ['o', 'ɛ', 'ᵻ', 'ɔ', 'æ', 'i', 'ɐ', 'ɜ', 'ə', 'ɑ', 'e', 'ʌ', 'ɚ', 'a', 'ɪ', 'ʊ', 'u']
    special_symbols = ['?', '.', '!', '~']

    plt.clf()
    plt.scatter(x=[x[0] for x in reduced_data], y=[x[1] for x in reduced_data], marker=MarkerStyle())
    plt.tight_layout()
    plt.axis('off')
    for index, phoneme in enumerate(reduced_data):
        x_position = phoneme[0]
        y_position = phoneme[1]
        label = phoneme_list[index]
        if label in special_symbols:
            color = "red"
        elif label in consonants:
            color = "blue"
        elif label in vowels:
            color = "green"
        else:
            color = "violet"
        plt.text(x=x_position, y=y_position, s=label, color=color)
    plt.subplots_adjust(top=0.85)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    with open("embedding_table_512dim.json", 'r', encoding="utf8") as fp:
        datapoints = json.load(fp)

    key_list = list()  # no matter where you get it from, this needs to be a list of the phonemes you want to visualize as string
    embedding_list = list()  # in the same order as the phonemes in the list above, this list needs to be filled with their embedding vectors

    for key in datapoints:
        key_list.append(key)
        embedding_list += datapoints[key]

    embeddings_as_array = np.array(embedding_list)

    tsne = TSNE(verbose=1, learning_rate=4, perplexity=30, n_iter=200000, n_iter_without_progress=8000, init='pca')
    pca = PCA(n_components=2)

    reduced_data_tsne = tsne.fit_transform(embeddings_as_array)
    reduced_data_pca = pca.fit_transform(embeddings_as_array)

    plot_embeddings(reduced_data_tsne, key_list, title="Trained Embeddings t-SNE")
    plot_embeddings(reduced_data_pca, key_list, title="Trained Embeddings PCA")
