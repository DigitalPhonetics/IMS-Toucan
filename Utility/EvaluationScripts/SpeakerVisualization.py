import numpy
import soundfile as sf
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle
from sklearn.manifold import TSNE
from tqdm import tqdm

from Preprocessing.ProsodicConditionExtractor import ProsodicConditionExtractor


class Visualizer:

    def __init__(self, sr=48000):
        """
        Args:
            sr: The sampling rate of the audios you want to visualize.
        """
        self.tsne = TSNE(verbose=1, learning_rate=4, perplexity=30, n_iter=200000, n_iter_without_progress=8000, init='pca')
        self.pros_cond_ext = ProsodicConditionExtractor(sr=sr)
        self.sr = sr

    def visualize_speaker_embeddings(self, label_to_filepaths, title_of_plot):
        label_list = list()
        embedding_list = list()
        for label in tqdm(label_to_filepaths):
            label_list.append(label)
            wave, sr = sf.read(label_to_filepaths[label])
            if self.sr != sr:
                print("One of the Audios you included doesn't match the sampling rate of this visualizer object, "
                      "creating a new condition extractor. Results will be correct, but if there are too many cases "
                      "of changing samplingrate, this will run very slowly.")
                self.pros_cond_ext = ProsodicConditionExtractor(sr=sr)
                self.sr = sr
            embedding_list.append(self.pros_cond_ext.extract_condition_from_reference_wave(wave).squeeze().numpy())
        embeddings_as_array = numpy.array(embedding_list)
        dimensionality_reduced_embeddings = self.tsne.fit_transform(embeddings_as_array)
        self._plot_embeddings(dimensionality_reduced_embeddings, label_list, title=title_of_plot)

    def _plot_embeddings(self, projected_data, labels, title):
        label_to_color = dict()
        for label in set(labels):
            label_to_color[label] = numpy.random.random()
        plt.clf()
        plt.scatter(x=[x[0] for x in projected_data],
                    y=[x[1] for x in projected_data],
                    marker=MarkerStyle(marker="."),
                    c=[label_to_color[label] for label in labels],
                    cmap='gist_rainbow')
        plt.tight_layout()
        plt.axis('off')
        plt.subplots_adjust(top=0.85)
        plt.title(title)
        plt.show()
