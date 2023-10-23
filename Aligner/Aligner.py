"""
taken and adapted from https://github.com/as-ideas/DeepForcedAligner
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn
from torch.nn import CTCLoss
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend


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
                 n_features=128,
                 num_symbols=145,
                 lstm_dim=512,
                 conv_dim=512):
        super().__init__()
        self.convs = nn.ModuleList([
            BatchNormConv(n_features, conv_dim, 3),
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
        self.proj = torch.nn.Linear(2 * lstm_dim, num_symbols)
        self.tf = ArticulatoryCombinedTextFrontend(language="en")
        self.ctc_loss = CTCLoss(blank=144, zero_infinity=True)
        self.vector_to_id = dict()

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

    @torch.inference_mode()
    def inference(self, features, tokens, save_img_for_debug=None, train=False, pathfinding="MAS", return_ctc=False):
        if not train:
            tokens_indexed = self.tf.text_vectors_to_id_sequence(text_vector=tokens)  # first we need to convert the articulatory vectors to IDs, so we can apply dijkstra or viterbi
            tokens = np.asarray(tokens_indexed)
        else:
            tokens = tokens.cpu().detach().numpy()

        pred = self(features.unsqueeze(0))
        if return_ctc:
            ctc_loss = self.ctc_loss(pred.transpose(0, 1).log_softmax(2), torch.LongTensor(tokens), torch.LongTensor([len(pred[0])]),
                                     torch.LongTensor([len(tokens)])).item()
        pred = pred.squeeze().cpu().detach().numpy()
        pred_max = pred[:, tokens]

        # run monotonic alignment search

        alignment_matrix = binarize_alignment(pred_max)

        if save_img_for_debug is not None:
            phones = list()
            for index in tokens:
                for phone in self.tf.phone_to_id:
                    if self.tf.phone_to_id[phone] == index:
                        phones.append(phone)
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

            ax.imshow(alignment_matrix, interpolation='nearest', aspect='auto', origin="lower", cmap='cividis')
            ax.set_ylabel("Mel-Frames")
            ax.set_xticks(range(len(pred_max[0])))
            ax.set_xticklabels(labels=phones)
            ax.set_title("MAS Path")

            plt.tight_layout()
            fig.savefig(save_img_for_debug)
            fig.clf()
            plt.close()

        if return_ctc:
            return alignment_matrix, ctc_loss
        return alignment_matrix



def binarize_alignment(alignment_prob):
    """
    # Implementation by:
    # https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/FastPitch/fastpitch/alignment.py
    # https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/FastPitch/fastpitch/attn_loss_function.py

    Binarizes alignment with MAS.
    """
    # assumes features x text
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


if __name__ == '__main__':
    tf = ArticulatoryCombinedTextFrontend(language="en")
    from Preprocessing.HiFiCodecAudioPreprocessor import CodecAudioPreprocessor

    cap = CodecAudioPreprocessor(input_sr=-1)
    dummy_codebook_indexes = torch.randint(low=0, high=1023, size=[9, 20])
    codebook_frames = cap.indexes_to_codec_frames(dummy_codebook_indexes)
    alignment = Aligner().inference(codebook_frames.transpose(0, 1), tokens=tf.string_to_tensor("Hello world"))
    print(alignment.shape)
    plt.imshow(alignment, origin="lower", cmap="GnBu")
    plt.show()
