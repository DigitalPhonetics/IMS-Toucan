"""
taken and adapted from https://github.com/as-ideas/DeepForcedAligner
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from Preprocessing.ArticulatoryCombinedTextFrontend import ArticulatoryCombinedTextFrontend


class BatchNormConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=1, padding=kernel_size // 2, bias=False)
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.relu(x)
        x = self.bnorm(x)
        x = x.transpose(1, 2)
        return x


class Aligner(torch.nn.Module):

    def __init__(self,
                 n_mels=80,
                 num_symbols=145,
                 lstm_dim=512,
                 conv_dim=512):
        super().__init__()
        self.convs = nn.ModuleList([
            BatchNormConv(n_mels, conv_dim, 3),
            nn.Dropout(p=0.5),
            BatchNormConv(conv_dim, conv_dim, 3),
            nn.Dropout(p=0.5),
            BatchNormConv(conv_dim, conv_dim, 3),
            nn.Dropout(p=0.5),
            ])
        self.rnn = torch.nn.LSTM(conv_dim, lstm_dim, batch_first=True, bidirectional=True)
        self.proj = torch.nn.Linear(2 * lstm_dim, num_symbols)
        self.tf = ArticulatoryCombinedTextFrontend(language="en")

    def forward(self, x, lens=None):
        for conv in self.convs:
            x = conv(x)

        if lens is not None:
            x = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.rnn(x)
        if lens is not None:
            x, _ = pad_packed_sequence(x, batch_first=True)

        x = self.proj(x)

        return x

    def inference(self, mel, tokens, save_img_for_debug=None, train=False, pathfinding="MAS"):

        if not train:
            tokens_indexed = list()  # first we need to convert the articulatory vectors to IDs, so we can apply dijkstra
            for vector in tokens:
                for phone in self.tf.phone_to_vector:
                    if vector.cpu().detach().numpy().tolist() == self.tf.phone_to_vector[phone]:
                        tokens_indexed.append(self.tf.phone_to_id[phone])
                        # this is terribly inefficient, but it's good enough for testing for now.
            tokens = torch.LongTensor(tokens_indexed).numpy()
        else:
            tokens = tokens.cpu().detach().numpy()

        pred = self(mel.unsqueeze(0)).squeeze().cpu().detach().numpy()
        pred_max = pred[:, tokens]
        path_probs = 1. - pred_max
        adj_matrix = to_adj_matrix(path_probs)

        if pathfinding == "MAS":

            alignment_matrix = binarize_alignment(pred_max)

            if save_img_for_debug is not None:
                phones = list()
                for index in tokens:
                    for phone in self.tf.phone_to_id:
                        if self.tf.phone_to_id[phone] == index:
                            phones.append(phone)
                fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 9))

                ax[0].imshow(pred_max, interpolation='nearest', aspect='auto', origin="lower")
                ax[1].imshow(alignment_matrix, interpolation='nearest', aspect='auto', origin="lower", cmap='cividis')

                ax[0].set_ylabel("Mel-Frames")
                ax[1].set_ylabel("Mel-Frames")

                ax[0].set_xticks(range(len(pred_max[0])))
                ax[0].set_xticklabels(labels=phones)

                ax[1].set_xticks(range(len(pred_max[0])))
                ax[1].set_xticklabels(labels=phones)

                ax[0].set_title("Path Probabilities")
                ax[1].set_title("MAS Path")

                plt.tight_layout()
                fig.savefig(save_img_for_debug)
                fig.clf()
                plt.close()

            return alignment_matrix

        elif pathfinding == "dijkstra":

            dist_matrix, predecessors, *_ = dijkstra(csgraph=adj_matrix,
                                                     directed=True,
                                                     indices=0,
                                                     return_predecessors=True)
            path = []
            pr_index = predecessors[-1]
            while pr_index != 0:
                path.append(pr_index)
                pr_index = predecessors[pr_index]
            path.reverse()

            # append first and last node
            path = [0] + path + [dist_matrix.size - 1]
            cols = path_probs.shape[1]
            mel_text = {}

            # collect indices (mel, text) along the path
            for node_index in path:
                i, j = from_node_index(node_index, cols)
                mel_text[i] = j

            path_plot = np.zeros_like(pred_max)
            for i in mel_text:
                path_plot[i][mel_text[i]] = 1.0

            if save_img_for_debug is not None:

                phones = list()
                for index in tokens:
                    for phone in self.tf.phone_to_id:
                        if self.tf.phone_to_id[phone] == index:
                            phones.append(phone)
                fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 9))

                ax[0].imshow(pred_max, interpolation='nearest', aspect='auto', origin="lower")
                ax[1].imshow(path_plot, interpolation='nearest', aspect='auto', origin="lower", cmap='cividis')

                ax[0].set_ylabel("Mel-Frames")
                ax[1].set_ylabel("Mel-Frames")

                ax[0].set_xticks(range(len(pred_max[0])))
                ax[0].set_xticklabels(labels=phones)

                ax[1].set_xticks(range(len(pred_max[0])))
                ax[1].set_xticklabels(labels=phones)

                ax[0].set_title("Path Probabilities")
                ax[1].set_title("Dijkstra Path")

                plt.tight_layout()
                fig.savefig(save_img_for_debug)
                fig.clf()
                plt.close()

            return path_plot


def binarize_alignment(alignment_prob):
    """
    # Implementation by:
    # https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/FastPitch/fastpitch/alignment.py
    # https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/FastPitch/fastpitch/attn_loss_function.py

    Binarizes alignment with MAS.
    """
    # assumes mel x text
    opt = np.zeros_like(alignment_prob)
    alignment_prob = alignment_prob + (np.abs(alignment_prob).max() + 1.0)  # make all numbers positive and add an offset to avoid log of 0 later
    alignment_prob * alignment_prob * (1.0 / alignment_prob.max())  # normalize to (0,  1]
    attn_map = np.log(alignment_prob)
    attn_map[0, 1:] = -np.inf
    log_p = np.zeros_like(attn_map)
    log_p[0, :] = attn_map[0, :]
    prev_ind = np.zeros_like(attn_map, dtype=np.int64)
    for i in range(1, attn_map.shape[0]):
        for j in range(attn_map.shape[1]):  # for each text dim
            prev_log = log_p[i - 1, j]
            prev_j = j
            if j - 1 >= 0 and log_p[i - 1, j - 1] >= log_p[i - 1, j]:
                prev_log = log_p[i - 1, j - 1]
                prev_j = j - 1
            log_p[i, j] = attn_map[i, j] + prev_log
            prev_ind[i, j] = prev_j
    # now backtrack
    curr_text_idx = attn_map.shape[1] - 1
    for i in range(attn_map.shape[0] - 1, -1, -1):
        opt[i, curr_text_idx] = 1
        curr_text_idx = prev_ind[i, curr_text_idx]
    opt[0, curr_text_idx] = 1
    return opt


def to_node_index(i, j, cols):
    return cols * i + j


def from_node_index(node_index, cols):
    return node_index // cols, node_index % cols


def to_adj_matrix(mat):
    rows = mat.shape[0]
    cols = mat.shape[1]

    row_ind = []
    col_ind = []
    data = []

    for i in range(rows):
        for j in range(cols):

            node = to_node_index(i, j, cols)

            if j < cols - 1:
                right_node = to_node_index(i, j + 1, cols)
                weight_right = mat[i, j + 1]
                row_ind.append(node)
                col_ind.append(right_node)
                data.append(weight_right)

            if i < rows - 1 and j < cols:
                bottom_node = to_node_index(i + 1, j, cols)
                weight_bottom = mat[i + 1, j]
                row_ind.append(node)
                col_ind.append(bottom_node)
                data.append(weight_bottom)

            if i < rows - 1 and j < cols - 1:
                bottom_right_node = to_node_index(i + 1, j + 1, cols)
                weight_bottom_right = mat[i + 1, j + 1]
                row_ind.append(node)
                col_ind.append(bottom_right_node)
                data.append(weight_bottom_right)

    adj_mat = coo_matrix((data, (row_ind, col_ind)), shape=(rows * cols, rows * cols))
    return adj_mat.tocsr()
