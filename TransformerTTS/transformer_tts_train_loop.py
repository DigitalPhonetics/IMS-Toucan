import json
import os
import time

import matplotlib.pyplot as plt
import torch
from adabound import AdaBound
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader


def plot_attentions(atts, dir, step):
    fig, axes = plt.subplots(nrows=len(atts) // 2, ncols=2, figsize=(6, 8))
    atts_1 = atts[::2]
    atts_2 = atts[1::2]
    for index, att in enumerate(atts_1):
        axes[index][0].imshow(att.detach().numpy(), cmap='BuPu_r', interpolation='nearest', aspect='auto',
                              origin="lower")
        axes[index][0].xaxis.set_visible(False)
        axes[index][0].yaxis.set_visible(False)
    for index, att in enumerate(atts_2):
        axes[index][1].imshow(att.detach().numpy(), cmap='BuPu_r', interpolation='nearest', aspect='auto',
                              origin="lower")
        axes[index][1].xaxis.set_visible(False)
        axes[index][1].yaxis.set_visible(False)
    plt.subplots_adjust(left=0.02, bottom=0.02, right=.98, top=.98, wspace=0, hspace=0)
    if not os.path.exists(os.path.join(dir, "atts")):
        os.makedirs(os.path.join(dir, "atts"))
    plt.savefig(os.path.join(os.path.join(dir, "atts"), str(step) + ".png"))
    plt.clf()
    plt.close()


def get_atts(model, lang, device):
    from PreprocessingForTTS.ProcessText import TextFrontend
    tf = TextFrontend(language=lang,
                      use_panphon_vectors=False,
                      use_sentence_type=False,
                      use_word_boundaries=False,
                      use_explicit_eos=True)
    sentence = "Hello"
    if lang == "en":
        sentence = "This is a brand new sentence."
    elif lang == "de":
        sentence = "Dies ist ein brandneuer Satz."
    atts = model.inference(tf.string_to_tensor(sentence).long().squeeze(0).to(device))[2].to("cpu")
    del tf
    return atts


def collate_and_pad(batch):
    if len(batch[0]) == 4:
        # every entry in batch: [text, text_length, spec, spec_length]
        texts = list()
        text_lens = list()
        speechs = list()
        speech_lens = list()
        for datapoint in batch:
            texts.append(torch.LongTensor(datapoint[0]).squeeze(0))
            text_lens.append(torch.LongTensor([datapoint[1]]))
            speechs.append(torch.Tensor(datapoint[2]))
            speech_lens.append(torch.LongTensor([datapoint[3]]))
        return (pad_sequence(texts, batch_first=True),
                torch.stack(text_lens).squeeze(1),
                pad_sequence(speechs, batch_first=True),
                torch.stack(speech_lens).squeeze(1))
    elif len(batch[0]) == 5:
        # every entry in batch: [text, text_length, spec, spec_length, spemb]
        texts = list()
        text_lens = list()
        speechs = list()
        speech_lens = list()
        spembs = list()
        for datapoint in batch:
            texts.append(torch.LongTensor(datapoint[0]).squeeze(0))
            text_lens.append(torch.LongTensor([datapoint[1]]))
            speechs.append(torch.Tensor(datapoint[2]))
            speech_lens.append(torch.LongTensor([datapoint[3]]))
            spembs.append(torch.Tensor(datapoint[4]))
        return (pad_sequence(texts, batch_first=True),
                torch.stack(text_lens).squeeze(1),
                pad_sequence(speechs, batch_first=True),
                torch.stack(speech_lens).squeeze(1),
                torch.stack(spembs))  # spembs may need squeezing


def train_loop(net, train_dataset, eval_dataset, device, save_directory,
               config, batchsize=10, epochs=150, gradient_accumulation=6,
               epochs_per_save=60, spemb=False, lang="en"):
    """
    :param lang: language for the sentence for attention plotting
    :param spemb: whether the dataset provides speaker embeddings
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
                              num_workers=16,
                              pin_memory=False,
                              shuffle=True,
                              prefetch_factor=16,
                              collate_fn=collate_and_pad,
                              persistent_workers=True)
    valid_loader = DataLoader(batch_size=50,  # this works perfectly as long as our eval set size is divisible by 50
                              dataset=eval_dataset,
                              drop_last=False,
                              num_workers=10,
                              pin_memory=False,
                              prefetch_factor=5,
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
                if not spemb:
                    train_loss = net(train_datapoint[0].to(device),
                                     train_datapoint[1].to(device),
                                     train_datapoint[2].to(device),
                                     train_datapoint[3].to(device))
                else:
                    train_loss = net(train_datapoint[0].to(device),
                                     train_datapoint[1].to(device),
                                     train_datapoint[2].to(device),
                                     train_datapoint[3].to(device),
                                     train_datapoint[4].to(device))
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
                if not spemb:
                    val_losses.append(float(net(validation_datapoint[0].to(device),
                                                validation_datapoint[1].to(device),
                                                validation_datapoint[2].to(device),
                                                validation_datapoint[3].to(device))))
                else:
                    val_losses.append(float(net(validation_datapoint[0].to(device),
                                                validation_datapoint[1].to(device),
                                                validation_datapoint[2].to(device),
                                                validation_datapoint[3].to(device),
                                                validation_datapoint[4].to(device))))
            average_val_loss = sum(val_losses) / len(val_losses)
            if epoch % epochs_per_save == 0:
                torch.save({"model": net.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scaler": scaler.state_dict()},
                           os.path.join(save_directory,
                                        "checkpoint_{}.pt".format(step_counter)))
                plot_attentions(torch.cat([att_w for att_w in get_atts(model=net, lang=lang, device=device)], dim=0),
                                dir=save_directory, step=step_counter)
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
