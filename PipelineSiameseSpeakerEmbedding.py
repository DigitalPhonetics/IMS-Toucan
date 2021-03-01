"""
Train a speaker embedding function on the vox celeb 2 corpus
"""

import json
import os
import time
import warnings

import torch
import torchviz
from adabound import AdaBound

from SpeakerEmbedding.SiameseSpeakerEmbedding import SiameseSpeakerEmbedding
from SpeakerEmbedding.SpeakerEmbeddingDataset import SpeakerEmbeddingDataset

warnings.filterwarnings("ignore")


def train_loop(net,
               train_dataset,
               valid_dataset,
               save_directory,
               device,
               epochs=1000,
               batchsize=32,
               steps_per_epoch=300,
               evaluation_samples=200):
    start_time = time.time()
    loss_plot = [[], []]
    with open(os.path.join(save_directory, "config.txt"), "w+") as conf:
        conf.write(net.get_conf())
    val_loss_highscore = 100.0
    batch_counter = 0
    net.train()
    net = net.to(device)
    optimizer = AdaBound(net.parameters())
    for epoch in range(epochs):
        train_losses = list()
        # train one epoch
        for _ in range(steps_per_epoch):
            for _ in range(batchsize):
                train_datapoint = next(train_dataset)
                train_loss = net(train_datapoint[0], train_datapoint[1], train_datapoint[2])
                train_losses.append(float(train_loss))
                (train_loss / batchsize).backward()  # for accumulative gradient
                batch_counter += 1
                if batch_counter % batchsize == 0:
                    print("Step: {}".format(int(batch_counter // batchsize)))
                    optimizer.step()
                    optimizer.zero_grad()
        # evaluate after epoch
        with torch.no_grad():
            net.eval()
            average_val_loss = 0
            average_train_loss = sum(train_losses) / len(train_losses)
            for _ in range(evaluation_samples):
                eval_datapoint = next(valid_dataset)
                average_val_loss += float(net(eval_datapoint[0],
                                              eval_datapoint[1],
                                              eval_datapoint[2]) / evaluation_samples)
            if val_loss_highscore > average_val_loss:
                val_loss_highscore = average_val_loss
                torch.save({"model": net.state_dict(),
                            "optimizer": optimizer.state_dict()},
                           os.path.join(save_directory, "checkpoint_{}.pt".format(round(average_val_loss, 4))))
            print("Epoch:        {}".format(epoch + 1))
            print("Train Loss:   {}".format(average_train_loss))
            print("Valid Loss:   {}".format(average_val_loss))
            print("Time elapsed: {} Minutes".format(round((time.time() - start_time) / 60), 2))
            loss_plot[0].append(average_train_loss)
            loss_plot[1].append(average_val_loss)
            with open(os.path.join(save_directory, "train_val_loss.json"), 'w') as fp:
                json.dump(loss_plot, fp)
            net.train()


def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def show_model(net=SiameseSpeakerEmbedding()):
    print(net)
    print("\n\nNumber of Parameters: {}".format(count_parameters(net)))


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
