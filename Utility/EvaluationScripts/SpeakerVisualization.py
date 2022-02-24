import matplotlib
import numpy
import soundfile as sf
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle

matplotlib.use("tkAgg")
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from tqdm import tqdm

from Preprocessing.ProsodicConditionExtractor import ProsodicConditionExtractor


class Visualizer:

    def __init__(self, sr=48000, device="cpu"):
        """
        Args:
            sr: The sampling rate of the audios you want to visualize.
        """
        self.tsne = TSNE(verbose=1, learning_rate=4, perplexity=30, n_iter=200000, n_iter_without_progress=8000, init='pca', n_jobs=-1)
        self.pca = PCA(n_components=2)
        self.pros_cond_ext = ProsodicConditionExtractor(sr=sr, device=device)
        self.sr = sr

    def visualize_speaker_embeddings(self, label_to_filepaths, title_of_plot, save_file_path=None):
        label_list = list()
        embedding_list = list()
        for label in tqdm(label_to_filepaths):
            for filepath in tqdm(label_to_filepaths[label]):
                wave, sr = sf.read(filepath)
                if len(wave) / sr < 1:
                    continue
                if self.sr != sr:
                    print("One of the Audios you included doesn't match the sampling rate of this visualizer object, "
                          "creating a new condition extractor. Results will be correct, but if there are too many cases "
                          "of changing samplingrate, this will run very slowly.")
                    self.pros_cond_ext = ProsodicConditionExtractor(sr=sr)
                    self.sr = sr
                embedding_list.append(self.pros_cond_ext.extract_condition_from_reference_wave(wave).squeeze().numpy())
                label_list.append(label)
        embeddings_as_array = numpy.array(embedding_list)

        dimensionality_reduced_embeddings_tsne = self.tsne.fit_transform(embeddings_as_array)
        dimensionality_reduced_embeddings_pca = self.pca.fit_transform(embeddings_as_array)

        self._plot_embeddings(dimensionality_reduced_embeddings_tsne, label_list, title=title_of_plot + " t-SNE", save_file_path=save_file_path)
        self._plot_embeddings(dimensionality_reduced_embeddings_pca, label_list, title=title_of_plot + " PCA", save_file_path=save_file_path)

    def _plot_embeddings(self, projected_data, labels, title, save_file_path):
        label_to_color = dict()
        for index, label in enumerate(list(set(labels))):
            label_to_color[label] = (1 / len(labels)) * index
        plt.clf()
        plt.scatter(x=[x[0] for x in projected_data],
                    y=[x[1] for x in projected_data],
                    marker=MarkerStyle(marker="."),
                    c=[label_to_color[label] for label in labels],
                    cmap='gist_rainbow',
                    alpha=0.6)
        plt.tight_layout()
        plt.axis('off')
        plt.subplots_adjust(top=0.9, bottom=0.0, right=1.0, left=0.0)
        plt.title(title)
        if save_file_path is not None:
            plt.savefig(save_file_path)
        else:
            plt.show()
        plt.close()
