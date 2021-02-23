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
from torch.nn.utils.rnn import pad_sequence
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
    # every entry in batch: [text, text_length, spec, spec_length]
    texts = list()
    text_lens = list()
    speechs = list()
    speech_lens = list()
    for datapoint in batch:
        texts.append(datapoint[0])
        text_lens.append(datapoint[1])
        speechs.append(datapoint[2])
        speech_lens.append(datapoint[3])
    return (pad_sequence(texts, batch_first=True),
            torch.stack(text_lens).squeeze(1),
            pad_sequence(speechs, batch_first=True),
            torch.stack(speech_lens).squeeze(1))


def train_loop(net, train_dataset, eval_dataset, device, save_directory,
               config, batchsize=10, epochs=150, gradient_accumulation=6):
    """
    :param net: Model to train
    :param train_dataset: Pytorch Dataset Object for train data
    :param eval_dataset: Pytorch Dataset Object for validation data
    :param device: Device to put the loaded tensors on
    :param save_directory: Where to save the checkpoints
    :param config: Config of the model to be trained
    :param batchsize: How many elements should be loaded at once
    :param epochs: how many epochs to train for
    :param gradient_accumulation: how many batches to average before stepping
    """
    train_loader = DataLoader(batch_size=batchsize,
                              dataset=train_dataset,
                              drop_last=True,
                              num_workers=batchsize,
                              pin_memory=True,
                              shuffle=True,
                              prefetch_factor=gradient_accumulation,
                              collate_fn=collate_and_pad)
    valid_loader = DataLoader(batch_size=1,
                              dataset=eval_dataset,
                              drop_last=False,
                              num_workers=2,
                              pin_memory=True,
                              prefetch_factor=2,
                              collate_fn=collate_and_pad)
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
        grad_accum = list()
        train_losses_this_epoch = list()
        for train_datapoint in train_loader:
            grad_accum.append(net(train_datapoint[0].to(device),
                                  train_datapoint[1].to(device),
                                  train_datapoint[2].to(device),
                                  train_datapoint[3].to(device)))
            train_losses_this_epoch.append(float(grad_accum[-1]))
            if len(grad_accum) % gradient_accumulation == 0:
                train_loss = 0.0
                for accum_loss in grad_accum:
                    train_loss += accum_loss
                train_loss /= len(grad_accum)
                grad_accum = list()
                optimizer.zero_grad()
                train_loss.backward()
                step_counter += 1
                # update weights
                print("Step: {}".format(step_counter))
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
        # evaluate on valid after every epoch is through
        with torch.no_grad():
            net.eval()
            val_losses = list()
            for validation_datapoint in valid_loader:
                val_losses.append(float(net(validation_datapoint[0].to(device),
                                            validation_datapoint[1].to(device),
                                            validation_datapoint[2].to(device),
                                            validation_datapoint[3].to(device))))
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
            print("Steps:        {}".format(step_counter))
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
    device = torch.device("cuda:3")
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
               batchsize=5,
               gradient_accumulation=12)
