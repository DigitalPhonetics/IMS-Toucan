import json
import os
import time

import torch
from adabound import AdaBound
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader


def collate_and_pad(batch):
    # every entry in batch: [text, text_length, spec, spec_length]
    texts = list()
    text_lens = list()
    speechs = list()
    speech_lens = list()
    for datapoint in batch:
        texts.append(torch.LongTensor(datapoint[0]))
        text_lens.append(torch.LongTensor([datapoint[1]]))
        speechs.append(torch.Tensor(datapoint[2]))
        speech_lens.append(torch.LongTensor([datapoint[3]]))
    return (pad_sequence(texts, batch_first=True),
            torch.stack(text_lens).squeeze(1),
            pad_sequence(speechs, batch_first=True),
            torch.stack(speech_lens).squeeze(1))


def train_loop(net, train_dataset, eval_dataset, device, save_directory,
               config, batchsize=10, epochs=150, gradient_accumulation=6,
               epochs_per_save=10):
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
    :param epochs_per_save: how many epochs to train in between checkpoints
    """
    net = net.to(device)
    scaler = GradScaler()
    train_loader = DataLoader(batch_size=batchsize,
                              dataset=train_dataset,
                              drop_last=True,
                              num_workers=4,
                              pin_memory=False,
                              shuffle=True,
                              prefetch_factor=2,
                              collate_fn=collate_and_pad,
                              persistent_workers=True)
    valid_loader = DataLoader(batch_size=1,
                              dataset=eval_dataset,
                              drop_last=False,
                              num_workers=2,
                              pin_memory=False,
                              prefetch_factor=2,
                              collate_fn=collate_and_pad,
                              persistent_workers=True)
    loss_plot = [[], []]
    with open(os.path.join(save_directory, "config.txt"), "w+") as conf:
        conf.write(config)
    step_counter = 0
    net.train()
    optimizer = AdaBound(net.parameters())
    start_time = time.time()
    for epoch in range(epochs):
        # train one epoch
        grad_accum = 0
        optimizer.zero_grad()
        train_losses_this_epoch = list()
        for train_datapoint in train_loader:
            with autocast():
                train_loss = net(train_datapoint[0].to(device),
                                 train_datapoint[1].to(device),
                                 train_datapoint[2].to(device),
                                 train_datapoint[3].to(device))
                train_losses_this_epoch.append(float(train_loss))
            scaler.scale((train_loss / gradient_accumulation)).backward()
            del train_loss
            grad_accum += 1
            if grad_accum % gradient_accumulation == 0:
                grad_accum = 0
                step_counter += 1
                # update weights
                # print("Step: {}".format(step_counter))
                scaler.step(optimizer)
                scaler.update()
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
            if epoch % epochs_per_save == 0:
                torch.save({"model": net.state_dict(),
                            "optimizer": optimizer.state_dict()},
                           os.path.join(save_directory,
                                        "checkpoint_{}.pt".format(step_counter)))
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
