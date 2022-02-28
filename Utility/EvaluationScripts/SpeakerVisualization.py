import matplotlib
import numpy
import soundfile as sf
from matplotlib import pyplot as plt

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
        self.tsne = TSNE(n_jobs=-1)
        self.pca = PCA(n_components=2)
        self.pros_cond_ext = ProsodicConditionExtractor(sr=sr, device=device)
        self.sr = sr

    def visualize_speaker_embeddings(self, label_to_filepaths, title_of_plot, save_file_path=None, include_pca=True, legend=True):
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
        self._plot_embeddings(projected_data=dimensionality_reduced_embeddings_tsne,
                              labels=label_list,
                              title=title_of_plot + " t-SNE" if include_pca else title_of_plot,
                              save_file_path=save_file_path,
                              legend=legend)

        if include_pca:
            dimensionality_reduced_embeddings_pca = self.pca.fit_transform(embeddings_as_array)
            self._plot_embeddings(projected_data=dimensionality_reduced_embeddings_pca,
                                  labels=label_list,
                                  title=title_of_plot + " PCA",
                                  save_file_path=save_file_path,
                                  legend=legend)

    def _plot_embeddings(self, projected_data, labels, title, save_file_path, legend):
        label_to_color = dict()
        for index, label in enumerate(list(set(labels))):
            label_to_color[label] = (1 / len(labels)) * index
        labels_to_points_x = dict()
        labels_to_points_y = dict()
        for label in labels:
            labels_to_points_x[label] = list()
            labels_to_points_y[label] = list()
        for index, label in enumerate(labels):
            labels_to_points_x[label].append(projected_data[index][0])
            labels_to_points_y[label].append(projected_data[index][1])

        fig, ax = plt.subplots()
        for label in set(labels):
            x = numpy.array(labels_to_points_x[label])
            y = numpy.array(labels_to_points_y[label])
            print(x.shape)
            print(label_to_color[label])
            ax.scatter(x=x,
                       y=y,
                       c=[label_to_color[label]] * len(x),
                       cmap='gist_rainbow',
                       label=label,
                       alpha=0.8)
        if legend:
            ax.legend()
        fig.tight_layout()
        ax.axis('off')
        fig.subplots_adjust(top=0.9, bottom=0.0, right=1.0, left=0.0)
        ax.set_title(title)
        if save_file_path is not None:
            plt.savefig(save_file_path)
        else:
            plt.show()
        plt.close()
