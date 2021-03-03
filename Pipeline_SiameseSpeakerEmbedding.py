"""
Train a speaker embedding function on the vox celeb 2 corpus
"""

import os
import warnings

import torch
import torchviz

from SpeakerEmbedding.SiameseSpeakerEmbedding import SiameseSpeakerEmbedding
from SpeakerEmbedding.SpeakerEmbeddingDataset import SpeakerEmbeddingDataset
from SpeakerEmbedding.speaker_embedding_train_loop import train_loop

warnings.filterwarnings("ignore")


def plot_model():
    sse = SiameseSpeakerEmbedding()
    out = sse(torch.rand((1, 1, 80, 2721)), torch.rand((1, 1, 80, 1233)), torch.Tensor([-1]))
    torchviz.make_dot(out.mean(), dict(sse.named_parameters())).render("speaker_emb_graph", format="png")


if __name__ == '__main__':

    print("Preparation")
    if not os.path.exists("Models"):
        os.mkdir("Models")
    if not os.path.exists("Models/SpeakerEmbedding"):
        os.mkdir("Models/SpeakerEmbedding")
    train_data = SpeakerEmbeddingDataset(train=True)
    valid_data = SpeakerEmbeddingDataset(train=False)

    print("Training")
    model = SiameseSpeakerEmbedding()
    train_loop(net=model,
               train_dataset=train_data,
               valid_dataset=valid_data,
               save_directory="Models/SpeakerEmbedding",
               device=torch.device("cpu"))
