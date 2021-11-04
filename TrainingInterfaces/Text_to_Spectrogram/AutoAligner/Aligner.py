"""
taken and adapted from https://github.com/as-ideas/DeepForcedAligner
"""

import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

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
                 n_mels,
                 num_symbols,
                 lstm_dim=256,
                 conv_dim=256):
        super().__init__()
        self.convs = nn.ModuleList([
            BatchNormConv(n_mels, conv_dim, 3),
            nn.Dropout(p=0.5),
            BatchNormConv(conv_dim, conv_dim, 3),
            nn.Dropout(p=0.5),
            BatchNormConv(conv_dim, conv_dim, 3),
            nn.Dropout(p=0.5),
            BatchNormConv(conv_dim, conv_dim, 3),
            nn.Dropout(p=0.5),
            BatchNormConv(conv_dim, conv_dim, 3),
            nn.Dropout(p=0.5),
        ])
        self.rnn = torch.nn.LSTM(conv_dim, lstm_dim, batch_first=True, bidirectional=True)
        self.lin = torch.nn.Linear(2 * lstm_dim, num_symbols)
        self.tf = ArticulatoryCombinedTextFrontend(language="en")

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x, _ = self.rnn(x)
        x = self.lin(x)
        return x

    def inference(self, mel, tokens):
        tokens_final = list()  # first we need to convert the articulatory vectors to IDs, so we can apply dijkstra
        for vector in tokens:
            for phone in self.tf.phone_to_vector:
                if vector == self.tf.phone_to_vector[phone]:
                    tokens_final.append(self.tf.phone_to_id[phone])
                    # this is terribly inefficient, but it's good enough for testing for now.
        tokens = torch.LongTensor(tokens_final)

        pred = self(mel)
        pred_max = pred[:, tokens]
        path_probs = 1. - pred_max
        adj_matrix = to_adj_matrix(path_probs)
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
        durations = np.zeros(tokens.shape[0], dtype=np.int32)

        # collect indices (mel, text) along the path
        for node_index in path:
            i, j = from_node_index(node_index, cols)
            mel_text[i] = j

        for j in mel_text.values():
            durations[j] += 1

        return durations


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
