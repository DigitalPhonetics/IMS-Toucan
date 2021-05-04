import os
import time

import librosa.display as lbd
import matplotlib.pyplot as plt
import torch
import torch.multiprocessing
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from PreprocessingForTTS.ProcessText import TextFrontend
from Utility.WarmupScheduler import WarmupScheduler


def plot_progress_spec(net, device, save_dir, step, lang, reference_speaker_embedding_for_plot):
    tf = TextFrontend(language=lang, use_panphon_vectors=False, use_word_boundaries=False, use_explicit_eos=False)
    sentence = "Hello"
    if lang == "en":
        sentence = "This is an unseen sentence."
    elif lang == "de":
        sentence = "Dies ist ein ungesehener Satz."
    text = tf.string_to_tensor(sentence).long().squeeze(0).to(device)
    spec = net.inference(text=text, speaker_embeddings=reference_speaker_embedding_for_plot).transpose(0, 1).to("cpu").numpy()
    if not os.path.exists(os.path.join(save_dir, "spec")):
        os.makedirs(os.path.join(save_dir, "spec"))
    fig, ax = plt.subplots(nrows=1, ncols=1)
    lbd.specshow(spec, ax=ax, sr=16000, cmap='GnBu', y_axis='mel', x_axis='time', hop_length=256)
    plt.savefig(os.path.join(os.path.join(save_dir, "spec"), str(step) + ".png"))
    plt.clf()
    plt.close()


def collate_and_pad(batch):
    if len(batch[0]) == 7:
        # every entry in batch: [text, text_length, spec, spec_length, durations, energy, pitch]
        texts = list()
        text_lens = list()
        speechs = list()
        speech_lens = list()
        durations = list()
        pitch = list()
        energy = list()
        for datapoint in batch:
            texts.append(torch.LongTensor(datapoint[0]).squeeze(0))
            text_lens.append(torch.LongTensor([datapoint[1]]))
            speechs.append(torch.Tensor(datapoint[2]))
            speech_lens.append(torch.LongTensor([datapoint[3]]))
            durations.append(torch.LongTensor(datapoint[4]))
            energy.append(torch.Tensor(datapoint[5]))
            pitch.append(torch.Tensor(datapoint[6]))
        return (
            pad_sequence(texts, batch_first=True), torch.stack(text_lens).squeeze(1), pad_sequence(speechs, batch_first=True),
            torch.stack(speech_lens).squeeze(1),
            pad_sequence(durations, batch_first=True), pad_sequence(pitch, batch_first=True), pad_sequence(energy, batch_first=True))
    elif len(batch[0]) == 8:
        # every entry in batch: [text, text_length, spec, spec_length, durations, energy, pitch, speaker_embedding]
        texts = list()
        text_lens = list()
        speechs = list()
        speech_lens = list()
        durations = list()
        pitch = list()
        energy = list()
        speaker_embeddings = list()
        for datapoint in batch:
            texts.append(torch.LongTensor(datapoint[0]).squeeze(0))
            text_lens.append(torch.LongTensor([datapoint[1]]))
            speechs.append(torch.Tensor(datapoint[2]))
            speech_lens.append(torch.LongTensor([datapoint[3]]))
            durations.append(torch.LongTensor(datapoint[4]))
            energy.append(torch.Tensor(datapoint[5]))
            pitch.append(torch.Tensor(datapoint[6]))
            speaker_embeddings.append(torch.Tensor(datapoint[7]))
        return (
            pad_sequence(texts, batch_first=True), torch.stack(text_lens).squeeze(1), pad_sequence(speechs, batch_first=True),
            torch.stack(speech_lens).squeeze(1),
            pad_sequence(durations, batch_first=True), pad_sequence(pitch, batch_first=True), pad_sequence(energy, batch_first=True),
            torch.stack(speaker_embeddings))


def train_loop(net,
               train_dataset,
               device,
               save_directory,
               batch_size=32,
               steps=400000,
               gradient_accumulation=1,
               epochs_per_save=10,
               use_speaker_embedding=False,
               lang="en",
               lr=0.1,
               warmup_steps=14000,
               path_to_checkpoint=None,
               fine_tune=False):
    """
    :param steps: How many steps to train
    :param lr: The initial learning rate for the optimiser
    :param warmup_steps: how many warmup steps for the warmup scheduler
    :param path_to_checkpoint: reloads a checkpoint to continue training from there
    :param fine_tune: whether to load everything from a checkpoint, or only the model parameters
    :param lang: language of the synthesis
    :param use_speaker_embedding: whether to expect speaker embeddings
    :param net: Model to train
    :param train_dataset: Pytorch Dataset Object for train data
    :param device: Device to put the loaded tensors on
    :param save_directory: Where to save the checkpoints
    :param batch_size: How many elements should be loaded at once
    :param gradient_accumulation: how many batches to average before stepping
    :param epochs_per_save: how many epochs to train in between checkpoints
    """
    net = net.to(device)
    scaler = GradScaler()
    if use_speaker_embedding:
        reference_speaker_embedding_for_plot = torch.Tensor(train_dataset[0][7]).to(device)
    else:
        reference_speaker_embedding_for_plot = None
    torch.multiprocessing.set_sharing_strategy('file_system')
    train_loader = DataLoader(batch_size=batch_size,
                              dataset=train_dataset,
                              drop_last=True,
                              num_workers=8,
                              pin_memory=False,
                              shuffle=True,
                              prefetch_factor=8,
                              collate_fn=collate_and_pad,
                              persistent_workers=True)
    step_counter = 0
    net.train()
    if fine_tune:
        lr = lr * 0.01
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = WarmupScheduler(optimizer, warmup_steps=warmup_steps)
    epoch = 0
    if path_to_checkpoint is not None:
        # careful when restarting, plotting data will be overwritten!
        check_dict = torch.load(path_to_checkpoint, map_location=device)
        net.load_state_dict(check_dict["model"])
        if not fine_tune:
            optimizer.load_state_dict(check_dict["optimizer"])
            scaler.load_state_dict(check_dict["scaler"])
            scheduler.load_state_dict(check_dict["scheduler"])
            if "step_counter" in check_dict:
                step_counter = check_dict["step_counter"]
            else:
                # legacy support
                step_counter = int(path_to_checkpoint.split(".")[0].split("_")[-1])

    start_time = time.time()
    while True:
        epoch += 1
        grad_accum = 0
        optimizer.zero_grad()
        train_losses_this_epoch = list()
        for train_datapoint in tqdm(train_loader):
            if gradient_accumulation == 1:
                with autocast():
                    if not use_speaker_embedding:
                        train_loss = net(train_datapoint[0].to(device), train_datapoint[1].to(device), train_datapoint[2].to(device),
                                         train_datapoint[3].to(device), train_datapoint[4].to(device), train_datapoint[5].to(device),
                                         train_datapoint[6].to(device))
                    else:
                        train_loss = net(train_datapoint[0].to(device), train_datapoint[1].to(device), train_datapoint[2].to(device),
                                         train_datapoint[3].to(device), train_datapoint[4].to(device), train_datapoint[5].to(device),
                                         train_datapoint[6].to(device), train_datapoint[7].to(device))
                    train_losses_this_epoch.append(float(train_loss))
                scaler.scale(train_loss).backward()
                del train_loss
                grad_accum = 0
                step_counter += 1
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
            else:
                if not use_speaker_embedding:
                    train_loss = net(train_datapoint[0].to(device), train_datapoint[1].to(device), train_datapoint[2].to(device), train_datapoint[3].to(device),
                                     train_datapoint[4].to(device), train_datapoint[5].to(device), train_datapoint[6].to(device))
                else:
                    train_loss = net(train_datapoint[0].to(device), train_datapoint[1].to(device), train_datapoint[2].to(device), train_datapoint[3].to(device),
                                     train_datapoint[4].to(device), train_datapoint[5].to(device), train_datapoint[6].to(device), train_datapoint[7].to(device))
                train_losses_this_epoch.append(float(train_loss))
                (train_loss / gradient_accumulation).backward()
                del train_loss
                grad_accum += 1
                if grad_accum % gradient_accumulation == 0:
                    grad_accum = 0
                    step_counter += 1
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
        with torch.no_grad():
            net.eval()
            if epoch % epochs_per_save == 0:
                torch.save({
                    "model": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "step_counter": step_counter,
                    "scheduler": scheduler.state_dict(),
                }, os.path.join(save_directory, "checkpoint_{}.pt".format(step_counter)))
                plot_progress_spec(net, device, save_dir=save_directory, step=step_counter, lang=lang,
                                   reference_speaker_embedding_for_plot=reference_speaker_embedding_for_plot)
                if step_counter > steps:
                    # DONE
                    return
            print("Epoch:        {}".format(epoch + 1))
            print("Train Loss:   {}".format(sum(train_losses_this_epoch) / len(train_losses_this_epoch)))
            print("Time elapsed: {} Minutes".format(round((time.time() - start_time) / 60), 2))
            print("Steps:        {}".format(step_counter))
            net.train()
