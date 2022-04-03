import matplotlib
import soundfile as sf
import torch
from matplotlib import cm
from matplotlib import pyplot as plt

matplotlib.use("tkAgg")
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy
from tqdm import tqdm

from Preprocessing.ProsodicConditionExtractor import ProsodicConditionExtractor


class Visualizer:

    def __init__(self, sr=48000, device="cpu"):
        """
        Args:
            sr: The sampling rate of the audios you want to visualize.
        """
        self.tsne = TSNE(n_jobs=-1, n_iter_without_progress=4000, n_iter=20000)
        self.pca = PCA(n_components=2)
        self.pros_cond_ext = ProsodicConditionExtractor(sr=sr, device=device)
        self.sr = sr

    def visualize_speaker_embeddings(self, label_to_filepaths, title_of_plot, save_file_path=None, include_pca=True, legend=True, colors=None):
        label_list = list()
        embedding_list = list()
        ordered_labels = sorted(list(label_to_filepaths.keys()))
        for label in tqdm(ordered_labels):
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
                              legend=legend,
                              colors=colors)

        if include_pca:
            dimensionality_reduced_embeddings_pca = self.pca.fit_transform(embeddings_as_array)
            self._plot_embeddings(projected_data=dimensionality_reduced_embeddings_pca,
                                  labels=label_list,
                                  title=title_of_plot + " PCA",
                                  save_file_path=save_file_path,
                                  legend=legend,
                                  colors=colors)

    def _plot_embeddings(self, projected_data, labels, title, save_file_path, legend, colors):
        if colors is None:
            colors = cm.gist_rainbow(numpy.linspace(0, 1, len(set(labels))))
        label_to_color = dict()
        for index, label in enumerate(sorted(list(set(labels)))):
            label_to_color[label] = colors[index]

        labels_to_points_x = dict()
        labels_to_points_y = dict()
        for label in labels:
            labels_to_points_x[label] = list()
            labels_to_points_y[label] = list()
        for index, label in enumerate(labels):
            labels_to_points_x[label].append(projected_data[index][0])
            labels_to_points_y[label].append(projected_data[index][1])

        fig, ax = plt.subplots()
        for label in sorted(list(set(labels))):
            x = numpy.array(labels_to_points_x[label])
            y = numpy.array(labels_to_points_y[label])
            ax.scatter(x=x,
                       y=y,
                       c=label_to_color[label],
                       label=label,
                       alpha=0.9)
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

    def calculate_spk_sim(self, reference_path, comparisons):
        embedding_list = list()
        for filepath in tqdm(comparisons):
            wave, sr = sf.read(filepath)
            if len(wave) / sr < 1:
                continue
            if self.sr != sr:
                print("One of the Audios you included doesn't match the sampling rate of this visualizer object, "
                      "creating a new condition extractor. Results will be correct, but if there are too many cases "
                      "of changing samplingrate, this will run very slowly.")
                self.pros_cond_ext = ProsodicConditionExtractor(sr=sr)
                self.sr = sr
            embedding_list.append(self.pros_cond_ext.extract_condition_from_reference_wave(wave).squeeze())

        wave, sr = sf.read(reference_path)
        if self.sr != sr:
            self.pros_cond_ext = ProsodicConditionExtractor(sr=sr)
            self.sr = sr
        reference_embedding = self.pros_cond_ext.extract_condition_from_reference_wave(wave).squeeze()

        sims = list()
        for comp_emb in embedding_list:
            sims.append(torch.cosine_similarity(reference_embedding, comp_emb, dim=0))

        return (sum(sims) / len(sims)).item(), numpy.std(sims)
