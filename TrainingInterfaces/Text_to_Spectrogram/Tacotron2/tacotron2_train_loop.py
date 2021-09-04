import os
import time

import matplotlib.pyplot as plt
import torch
import torch.multiprocessing
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from Preprocessing.ArticulatoryCombinedTextFrontend import ArticulatoryCombinedTextFrontend
from Utility.utils import delete_old_checkpoints


def plot_attention(model, lang, device, speaker_embedding, att_dir, step):
    tf = ArticulatoryCombinedTextFrontend(language=lang)
    sentence = ""
    if lang == "en":
        sentence = "This is a complex sentence, it even has a pause!"
    elif lang == "de":
        sentence = "Dies ist ein komplexer Satz, er hat sogar eine Pause!"
    text = tf.string_to_tensor(sentence).to(device)
    phones = tf.get_phone_string(sentence)
    model.eval()
    att = model.inference(text_tensor=text, speaker_embeddings=speaker_embedding)[2].to("cpu")
    model.train()
    del tf
    plt.figure(figsize=(8, 4))
    plt.imshow(att.detach().numpy(), interpolation='nearest', aspect='auto', origin="lower")
    plt.xlabel("Inputs")
    plt.ylabel("Outputs")
    plt.xticks(range(len(att[0])), labels=phones)
    plt.tight_layout()
    if not os.path.exists(os.path.join(att_dir, "attention_plots")):
        os.makedirs(os.path.join(att_dir, "attention_plots"))
    plt.savefig(os.path.join(os.path.join(att_dir, "attention_plots"), str(step) + ".png"))
    plt.clf()
    plt.close()


def collate_and_pad(batch):
    if len(batch[0]) == 4:
        # every entry in batch: [text, text_length, spec, spec_length]
        texts = list()
        text_lens = list()
        speechs = list()
        speech_lens = list()
        for datapoint in batch:
            texts.append(torch.Tensor(datapoint[0]))
            text_lens.append(torch.LongTensor([datapoint[1]]))
            speechs.append(torch.Tensor(datapoint[2]))
            speech_lens.append(torch.LongTensor([datapoint[3]]))
        return (pad_sequence(texts, batch_first=True),
                torch.stack(text_lens).squeeze(1),
                pad_sequence(speechs, batch_first=True),
                torch.stack(speech_lens).squeeze(1))
    elif len(batch[0]) == 5:
        # every entry in batch: [text, text_length, spec, spec_length, speaker_embedding]
        texts = list()
        text_lens = list()
        speechs = list()
        speech_lens = list()
        speaker_embeddings = list()
        for datapoint in batch:
            texts.append(torch.Tensor(datapoint[0]))
            text_lens.append(torch.LongTensor([datapoint[1]]))
            speechs.append(torch.Tensor(datapoint[2]))
            speech_lens.append(torch.LongTensor([datapoint[3]]))
            speaker_embeddings.append(torch.Tensor(datapoint[4]))
        return (pad_sequence(texts, batch_first=True),
                torch.stack(text_lens).squeeze(1),
                pad_sequence(speechs, batch_first=True),
                torch.stack(speech_lens).squeeze(1),
                torch.stack(speaker_embeddings))


def train_loop(net,
               train_dataset,
               device,
               save_directory,
               batch_size=22,
               steps=100000,
               epochs_per_save=2,
               use_speaker_embedding=False,
               lang="en",
               lr=0.001,
               path_to_checkpoint=None,
               fine_tune=False):
    """
    :param steps: How many steps to train
    :param lr: The initial learning rate for the optimiser
    :param path_to_checkpoint: reloads a checkpoint to continue training from there
    :param fine_tune: whether to load everything from a checkpoint, or only the model parameters
    :param lang: language of the synthesis
    :param use_speaker_embedding: whether to expect speaker embeddings
    :param net: Model to train
    :param train_dataset: Pytorch Dataset Object for train data
    :param device: Device to put the loaded tensors on
    :param save_directory: Where to save the checkpoints
    :param batch_size: How many elements should be loaded at once
    :param epochs_per_save: how many epochs to train in between checkpoints
    """
    net = net.to(device)
    train_loader = DataLoader(batch_size=batch_size,
                              dataset=train_dataset,
                              drop_last=True,
                              num_workers=8,
                              pin_memory=False,
                              shuffle=True,
                              prefetch_factor=8,
                              collate_fn=collate_and_pad,
                              persistent_workers=True)
    if use_speaker_embedding:
        reference_speaker_embedding_for_att_plot = torch.Tensor(train_dataset[0][4]).to(device)
    else:
        reference_speaker_embedding_for_att_plot = None
    step_counter = 0
    epoch = 0
    net.train()
    if fine_tune:
        lr = lr * 0.01
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1.0e-06, weight_decay=0.0)
    scaler = GradScaler()
    if path_to_checkpoint is not None:
        check_dict = torch.load(os.path.join(path_to_checkpoint), map_location=device)
        net.load_state_dict(check_dict["model"])
        if not fine_tune:
            optimizer.load_state_dict(check_dict["optimizer"])
            step_counter = check_dict["step_counter"]
            scaler.load_state_dict(check_dict["scaler"])
    start_time = time.time()
    while True:
        epoch += 1
        optimizer.zero_grad()
        train_losses_this_epoch = list()
        for batch in tqdm(train_loader):
            with autocast():
                if not use_speaker_embedding:
                    train_loss = net(text=batch[0].to(device),
                                     text_lengths=batch[1].to(device),
                                     speech=batch[2].to(device),
                                     speech_lengths=batch[3].to(device),
                                     speaker_embeddings=None,
                                     step=step_counter)
                else:
                    train_loss = net(text=batch[0].to(device),
                                     text_lengths=batch[1].to(device),
                                     speech=batch[2].to(device),
                                     speech_lengths=batch[3].to(device),
                                     speaker_embeddings=batch[4].to(device),
                                     step=step_counter)
                train_losses_this_epoch.append(float(train_loss))
            optimizer.zero_grad()
            scaler.scale(train_loss).backward()
            step_counter += 1
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        net.eval()
        if epoch % epochs_per_save == 0:
            torch.save({
                "model"       : net.state_dict(),
                "optimizer"   : optimizer.state_dict(),
                "scaler"      : scaler.state_dict(),
                "step_counter": step_counter,
                }, os.path.join(save_directory, "checkpoint_{}.pt".format(step_counter)))
            delete_old_checkpoints(save_directory, keep=5)
            with torch.no_grad():
                plot_attention(model=net,
                               lang=lang,
                               device=device,
                               speaker_embedding=reference_speaker_embedding_for_att_plot,
                               att_dir=save_directory,
                               step=step_counter)
            if step_counter > steps:
                # DONE
                return
        print("Epoch:        {}".format(epoch + 1))
        print("Train Loss:   {}".format(sum(train_losses_this_epoch) / len(train_losses_this_epoch)))
        print("Time elapsed: {} Minutes".format(round((time.time() - start_time) / 60), 2))
        print("Steps:        {}".format(step_counter))
        net.train()
