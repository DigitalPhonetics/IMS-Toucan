import json

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from Preprocessing.ArticulatoryCombinedTextFrontend import ArticulatoryCombinedTextFrontend


def plot_embeddings(reduced_data, phoneme_list, title, save=False):
    consonants = ['w', 'b', 'ɡ', 'n', 'ʒ', 'ʃ', 'd', 'l', 'θ', 'ŋ', 'f', 'ɾ', 's', 'm', 't', 'h', 'z', 'p', 'ʔ', 'v', 'ɹ', 'j', 'ð', 'k']
    vowels = ['o', 'ɛ', 'ᵻ', 'ɔ', 'æ', 'i', 'ɐ', 'ɜ', 'ə', 'ɑ', 'e', 'ʌ', 'ɚ', 'a', 'ɪ', 'ʊ', 'u']
    special_symbols = ['?', '.', '!', '~', '#']
    uniques_v = ['y', 'ʏ', 'ø', 'œ', 'ε']
    uniques_c = ['ç', 'x']

    plt.clf()
    fig, ax = plt.subplots(ncols=1, nrows=1)
    fig.set_size_inches(3, 3)

    ax.scatter(x=[x[0] for x in reduced_data], y=[x[1] for x in reduced_data], marker=MarkerStyle())
    ax.axis('off')
    for index, phoneme in enumerate(reduced_data):
        x_position = phoneme[0]
        y_position = phoneme[1]
        label = phoneme_list[index]
        if label in special_symbols:
            color = "gray"
        elif label in consonants:
            color = "blue"
        elif label in vowels:
            color = "darkgreen"
        elif label in uniques_v:
            color = "darkorange"
        elif label in uniques_c:
            color = "darkred"
        else:
            continue
        ax.text(x=x_position, y=y_position, s=label, color=color)
    if not save:
        ax.title(title)
        plt.show()
    else:
        fig.subplots_adjust(top=1.0, bottom=0.0, left=0.0, right=1.0)
        fig.savefig(f"{title}.pdf")


if __name__ == '__main__':

    key_list = list()  # no matter where you get it from, this needs to be a list of the phonemes you want to visualize as string
    embedding_list = list()  # in the same order as the phonemes in the list above, this list needs to be filled with their embedding vectors

    text2phone = ArticulatoryCombinedTextFrontend(language="de", inference=True)

    phone_to_embedding = dict()
    for phone in text2phone.phone_to_vector:
        phone_to_embedding[phone] = text2phone.phone_to_vector[phone]

    for key in phone_to_embedding:
        key_list.append(key)
        embedding_list += [phone_to_embedding[key]]

    embeddings_as_array = np.array(embedding_list)

    tsne = TSNE(verbose=1, learning_rate=4, perplexity=30, n_iter=200000, n_iter_without_progress=8000, init='pca')
    pca = PCA(n_components=2)

    reduced_data_tsne = tsne.fit_transform(embeddings_as_array)
    reduced_data_pca = pca.fit_transform(embeddings_as_array)

    plot_embeddings(reduced_data_tsne, key_list, title="featurespace", save=True)

    ##########################################################################################################################
    with open("embedding_table_512dim.json", 'r', encoding="utf8") as fp:
        datapoints = json.load(fp)

    key_list = list()
    embedding_list = list()

    for key in datapoints:
        key_list.append(key)
        embedding_list += [datapoints[key]]

    embeddings_as_array = np.array(embedding_list)

    tsne = TSNE(verbose=1, learning_rate=4, perplexity=30, n_iter=200000, n_iter_without_progress=8000, init='pca')
    pca = PCA(n_components=2)

    reduced_data_tsne = tsne.fit_transform(embeddings_as_array)
    reduced_data_pca = pca.fit_transform(embeddings_as_array)

    plot_embeddings(reduced_data_tsne, key_list, title="embeddingspace_taco", save=True)

    ##########################################################################################################################
    with open("embedding_table_384dim.json", 'r', encoding="utf8") as fp:
        datapoints = json.load(fp)

    key_list = list()
    embedding_list = list()

    for key in datapoints:
        key_list.append(key)
        embedding_list += [datapoints[key]]

    embeddings_as_array = np.array(embedding_list)

    tsne = TSNE(verbose=1, learning_rate=4, perplexity=30, n_iter=200000, n_iter_without_progress=8000, init='pca')
    pca = PCA(n_components=2)

    reduced_data_tsne = tsne.fit_transform(embeddings_as_array)
    reduced_data_pca = pca.fit_transform(embeddings_as_array)

    plot_embeddings(reduced_data_tsne, key_list, title="embeddingspace_fast", save=True)
    # plot_embeddings(reduced_data_pca, key_list, title="Trained Embeddings PCA")
