"""
Train an autoregressive Transformer TTS model on the german single speaker dataset by Hokuspokus
"""

import json
import os
import random
import time

import torch
import torchviz

from TransformerTTS.TransformerTTS import Transformer
from TransformerTTS.TransformerTTSDataset import TransformerTTSDataset

torch.manual_seed(17)
random.seed(17)


def train_loop(net,
               train_dataset,
               eval_dataset,
               device,
               save_directory,
               config,
               epochs=150,
               samples_per_update=64):
    start_time = time.time()
    loss_plot = [[], []]
    with open(os.path.join(save_directory, "config.txt"), "w+") as conf:
        conf.write(config)
    val_loss_highscore = 100.0
    sample_counter = 0
    net = net.to(device)
    net.train()
    optimizer = torch.optim.Adam(net.parameters())
    for epoch in range(epochs):
        # train one epoch
        optimizer.zero_grad()
        # ^ get rid of remaining gradients from
        # previous epoch if samples per update
        # and element in dataset don't line up
        index_list = random.sample(range(len(train_dataset)), len(train_dataset))
        train_losses_this_epoch = list()
        for count, index in enumerate(index_list):
            # accumulate averaged gradient
            train_datapoint = train_dataset[index]
            train_loss = net(train_datapoint[0].unsqueeze(0).to(device),
                             train_datapoint[1].to(device),
                             train_datapoint[2].unsqueeze(0).to(device),
                             train_datapoint[3].to(device)
                             )[0]
            train_losses_this_epoch.append(float(train_loss))
            (train_loss / samples_per_update).backward()
            torch.cuda.empty_cache()
            sample_counter += 1
            if count % samples_per_update == 0 and count != 0:
                # update weights
                print("Sample: {}".format(sample_counter))
                optimizer.step()
                optimizer.zero_grad()
        # evaluate on valid after every epoch
        with torch.no_grad():
            net.eval()
            val_losses = list()
            for validation_datapoint_index in range(len(eval_dataset)):
                eval_datapoint = eval_dataset[validation_datapoint_index]
                val_losses.append(net(eval_datapoint[0].unsqueeze(0).to(device),
                                      eval_datapoint[1].to(device),
                                      eval_datapoint[2].unsqueeze(0).to(device),
                                      eval_datapoint[3].to(device)
                                      )[0])
            val_loss = float(sum(val_losses) / len(val_losses))
            if val_loss_highscore > val_loss:
                val_loss_highscore = val_loss
                torch.save({"model": net.state_dict(),
                            "optimizer": optimizer.state_dict()},
                           os.path.join(save_directory,
                                        "checkpoint_{}_{}.pt".format(round(val_loss, 4), sample_counter)))
            print("Epoch:        {}".format(epoch + 1))
            print("Train Loss:   {}".format(sum(train_losses_this_epoch) / len(train_losses_this_epoch)))
            print("Valid Loss:   {}".format(val_loss))
            print("Time elapsed: {} Minutes".format(round((time.time() - start_time) / 60), 2))
            loss_plot[0].append(sum(train_losses_this_epoch) / len(train_losses_this_epoch))
            loss_plot[1].append(val_loss)
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
    print("Extracting features")
    # fe = CSS10SingleSpeakerFeaturizer()
    # fe.featurize_corpus()
    print("Loading data")
    device = torch.device("cuda:2")
    with open("Corpora/TransformerTTS/SingleSpeaker/CSS10/features.json", 'r') as fp:
        feature_list = json.load(fp)
    print("Building datasets")
    css10_train = TransformerTTSDataset(feature_list, type="train")
    css10_valid = TransformerTTSDataset(feature_list, type="valid")
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
               epochs=600,
               samples_per_update=64)
