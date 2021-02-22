"""
Train an autoregressive Transformer TTS model on the german single speaker dataset by Hokuspokus
"""

import json
import os
import random
import time
import warnings

import torch
import torchviz
from adabound import AdaBound
from torch.utils.data.dataloader import DataLoader

from TransformerTTS.TransformerTTS import Transformer
from TransformerTTS.TransformerTTSDataset import TransformerTTSDataset

warnings.filterwarnings("ignore")

torch.manual_seed(17)
random.seed(17)


def build_path_to_transcript_dict():
    path_to_transcript = dict()
    with open("Corpora/CSS10/transcript.txt", encoding="utf8") as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            path_to_transcript[line.split("|")[0]] = line.split("|")[2]
    return path_to_transcript


def collate_and_pad(batch):
    print(batch)
    # return torch.stack(batch)
    return None


def train_loop(net,
               train_dataset,
               eval_dataset,
               device,
               save_directory,
               config,
               batchsize=64,
               epochs=150):
    train_loader = DataLoader(batch_size=batchsize, dataset=train_dataset, drop_last=True, num_workers=16,
                              pin_memory=True, prefetch_factor=4, collate_fn=collate_and_pad)
    valid_loader = DataLoader(batch_size=1, dataset=eval_dataset, drop_last=False, num_workers=4,
                              pin_memory=True, prefetch_factor=2, collate_fn=collate_and_pad)
    loss_plot = [[], []]
    with open(os.path.join(save_directory, "config.txt"), "w+") as conf:
        conf.write(config)
    val_loss_highscore = 100.0
    step_counter = 0
    net = net.to(device)
    net.train()
    optimizer = AdaBound(net.parameters())
    start_time = time.time()
    for epoch in range(epochs):
        # train one epoch
        train_losses_this_epoch = list()
        for train_datapoint in train_loader:
            train_loss = net(train_datapoint[0].unsqueeze(0).to(device),
                             train_datapoint[1].to(device),
                             train_datapoint[2].unsqueeze(0).to(device),
                             train_datapoint[3].to(device)
                             )[0]
            train_losses_this_epoch.append(float(train_loss))
            optimizer.zero_grad()
            train_loss.backward()
            step_counter += 1
            # update weights
            print("Step: {}".format(step_counter))
            optimizer.step()
        # evaluate on valid after every epoch is through
        with torch.no_grad():
            net.eval()
            val_losses = list()
            for validation_datapoint in valid_loader:
                val_losses.append(float(net(validation_datapoint[0].unsqueeze(0).to(device),
                                            validation_datapoint[1].to(device),
                                            validation_datapoint[2].unsqueeze(0).to(device),
                                            validation_datapoint[3].to(device)
                                            )[0]))
            average_val_loss = sum(val_losses) / len(val_losses)
            if val_loss_highscore > average_val_loss:
                val_loss_highscore = average_val_loss
                torch.save({"model": net.state_dict(),
                            "optimizer": optimizer.state_dict()},
                           os.path.join(save_directory,
                                        "checkpoint_{}_{}.pt".format(round(average_val_loss, 4), step_counter)))
            print("Epoch:        {}".format(epoch + 1))
            print("Train Loss:   {}".format(sum(train_losses_this_epoch) / len(train_losses_this_epoch)))
            print("Valid Loss:   {}".format(average_val_loss))
            print("Time elapsed: {} Minutes".format(round((time.time() - start_time) / 60), 2))
            loss_plot[0].append(sum(train_losses_this_epoch) / len(train_losses_this_epoch))
            loss_plot[1].append(average_val_loss)
            with open(os.path.join(save_directory, "train_val_loss.json"), 'w') as plotting_data_file:
                json.dump(loss_plot, plotting_data_file)
            net.train()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def show_model(model):
    print(model)
    print("\n\nNumber of Parameters: {}".format(count_parameters(model)))


def plot_model():
    trans = Transformer(idim=132, odim=80, spk_embed_dim=128)
    out = trans(text=torch.randint(high=120, size=(1, 23)),
                text_lengths=torch.tensor([23]),
                speech=torch.rand((1, 1234, 80)),
                speech_lengths=torch.tensor([1234]),
                spembs=torch.rand(128).unsqueeze(0))
    torchviz.make_dot(out[0].mean(), dict(trans.named_parameters())).render("transformertts_graph", format="png")


if __name__ == '__main__':
    print("Preparing")
    device = torch.device("cuda:2")
    path_to_transcript_dict = build_path_to_transcript_dict()
    css10_train = TransformerTTSDataset(path_to_transcript_dict, train=True)
    css10_valid = TransformerTTSDataset(path_to_transcript_dict, train=False)
    model = Transformer(idim=132, odim=80, spk_embed_dim=None)
    if not os.path.exists("Models/TransformerTTS/SingleSpeaker/CSS10"):
        os.makedirs("Models/TransformerTTS/SingleSpeaker/CSS10")
    print("Training model")
    train_loop(net=model,
               train_dataset=css10_train,
               eval_dataset=css10_valid,
               device=device,
               config=model.get_conf(),
               save_directory="Models/TransformerTTS/SingleSpeaker/CSS10",
               epochs=3000,  # just kill the process at some point
               batchsize=64)
