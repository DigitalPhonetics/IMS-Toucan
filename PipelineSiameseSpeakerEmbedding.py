import os
import random
import time

import torch
import torchviz

from PreprocessingForTTS.ProcessAudio import AudioPreprocessor
from SpeakerEmbedding.SiameseSpeakerEmbedding import SiameseSpeakerEmbedding
from SpeakerEmbedding.SpeakerEmbeddingDataset import SpeakerEmbeddingDataset


def featurize_corpus(path_to_corpus):
    # make a dict with keys being speakers and values being lists of all their utterances as melspec matrices
    # then dump this as json
    ap = AudioPreprocessor(input_sr=16000, melspec_buckets=512)
    pass


def train_loop(net, train_dataset, eval_dataset, save_directory, epochs=100, batchsize=64, device="cuda"):
    start_time = time.time()
    with open(os.path.join(save_directory, "config.txt"), "w+") as conf:
        conf.write(net.get_conf())
    val_loss_highscore = 100.0
    batch_counter = 0
    net.train()
    net.to_device(device)
    optimizer = torch.optim.Adam(net.parameters())
    for epoch in range(epochs):
        index_list = random.sample(range(len(train_dataset)), len(train_dataset))
        train_losses = list()
        # train one epoch
        for index in index_list:
            train_loss = net(train_dataset[index])[0]
            train_losses.append(train_loss / batchsize)  # for accumulative gradient
            train_losses[-1].backward()
            batch_counter += 1
            if batch_counter % batchsize == 0:
                print("Step:         {}".format(batch_counter))
                optimizer.step()
                optimizer.zero_grad()
        # evaluate after epoch
        with torch.no_grad():
            net.eval()
            val_losses = list()
            val_indexes = range(len(eval_dataset))
            for validation_datapoint_index in val_indexes:
                val_losses.append(net(eval_dataset[validation_datapoint_index])[0])
            val_loss = sum(val_losses) / len(val_losses)
            if val_loss_highscore > val_loss:
                val_loss_highscore = val_loss
                torch.save({"model": net.state_dict(),
                            "optimizer": optimizer.state_dict()},
                           save_directory / "checkpoint_{}.pt".format(round(val_loss, 4)))
            print("Epoch:        {}".format(epoch))
            print("Train Loss:   {}".format(sum(train_losses)))
            print("Valid Loss:   {}".format(val_loss))
            print("Time elapsed: {}".format(start_time - time.time()))
            net.train()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def show_model(model):
    print(model)
    print("\n\nNumber of Parameters: {}".format(count_parameters(model)))


def plot_model():
    sse = SiameseSpeakerEmbedding()
    out = sse(torch.rand((1, 1, 512, 2721)), torch.rand((1, 1, 512, 1233)), torch.Tensor([-1]))
    torchviz.make_dot(out.mean(), dict(sse.named_parameters())).render("speaker_emb_graph", format="png")


if __name__ == '__main__':
    print("Stage 1: Preparation")
    if not os.path.exists("Corpora"):
        os.mkdir("Corpora")
        if not os.path.exists("Corpora/SpeakerEmbedding"):
            os.mkdir("Corpora/SpeakerEmbedding")
    if not os.path.exists("Models"):
        os.mkdir("Models")
        if not os.path.exists("Models/SpeakerEmbedding"):
            os.mkdir("Models/SpeakerEmbedding")
    path_to_feature_dump_train = "Corpora/SpeakerEmbedding/train.json"
    path_to_feature_dump_valid = "Corpora/SpeakerEmbedding/valid.json"

    print("Stage 2: Feature Extraction")
    featurize_corpus(path_to_feature_dump_train)
    featurize_corpus(path_to_feature_dump_valid)

    print("Stage 3: Data Loading")
    train_data = SpeakerEmbeddingDataset(path_to_feature_dump_train)
    valid_data = SpeakerEmbeddingDataset(path_to_feature_dump_valid)

    print("Stage 4: Model Training")
    net = SiameseSpeakerEmbedding()
    train_loop(net, train_data, valid_data, "Models/SpeakerEmbedding")
