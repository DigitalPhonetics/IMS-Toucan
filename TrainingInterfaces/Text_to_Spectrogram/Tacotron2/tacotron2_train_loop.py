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

from Preprocessing.TextFrontend import TextFrontend
from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.AlignmentLoss import binarize_attention_parallel
from Utility.utils import delete_old_checkpoints
from Utility.utils import get_most_recent_checkpoint


def plot_attention(model, lang, device, speaker_embedding, att_dir, step):
    tf = TextFrontend(language=lang, use_word_boundaries=False, use_explicit_eos=False)
    sentence = ""
    if lang == "en":
        sentence = "This is a complex sentence, it even has a pause!"
    elif lang == "de":
        sentence = "Dies ist ein komplexer Satz, er hat sogar eine Pause!"
    text = tf.string_to_tensor(sentence).long().squeeze(0).to(device)
    phones = tf.get_phone_string(sentence)
    model.eval()
    att = model.inference(text=text, speaker_embeddings=speaker_embedding)[2].to("cpu")
    model.train()
    del tf
    bin_att = binarize_attention_parallel(att.unsqueeze(0).unsqueeze(1),
                                          in_lens=torch.LongTensor([len(text)]),
                                          out_lens=torch.LongTensor([len(att)])).squeeze(0).squeeze(0).detach().numpy()
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 9))
    ax[0].imshow(att.detach().numpy(), interpolation='nearest', aspect='auto', origin="lower")
    ax[1].imshow(bin_att, interpolation='nearest', aspect='auto', origin="lower")
    ax[1].set_xlabel("Inputs")
    ax[0].xaxis.set_visible(False)
    ax[0].set_ylabel("Outputs")
    ax[1].set_ylabel("Outputs")
    ax[1].set_xticks(range(len(att[0])))
    del att
    ax[1].set_xticklabels(labels=[phone for phone in phones])
    ax[0].set_title("Soft-Attention")
    ax[1].set_title("Hard-Attention")
    fig.tight_layout()
    if not os.path.exists(os.path.join(att_dir, "attention_plots")):
        os.makedirs(os.path.join(att_dir, "attention_plots"))
    fig.savefig(os.path.join(os.path.join(att_dir, "attention_plots"), str(step) + ".png"))
    fig.clf()
    plt.close()


def collate_and_pad(batch):
    if len(batch[0]) == 4:
        # every entry in batch: [text, text_length, spec, spec_length]
        return (pad_sequence([datapoint[0].squeeze(0) for datapoint in batch], batch_first=True),
                torch.stack([datapoint[1] for datapoint in batch]).squeeze(1),
                pad_sequence([datapoint[2] for datapoint in batch], batch_first=True),
                torch.stack([datapoint[3] for datapoint in batch]).squeeze(1))
    elif len(batch[0]) == 5:
        # every entry in batch: [text, text_length, spec, spec_length, speaker_embedding]
        return (pad_sequence([datapoint[0].squeeze(0) for datapoint in batch], batch_first=True),
                torch.stack([datapoint[1] for datapoint in batch]).squeeze(1),
                pad_sequence([datapoint[2] for datapoint in batch], batch_first=True),
                torch.stack([datapoint[3] for datapoint in batch]).squeeze(1),
                torch.stack([datapoint[4] for datapoint in batch]))


def train_loop(net,
               train_dataset,
               device,
               save_directory,
               batch_size,
               steps,
               epochs_per_save,
               lang,
               lr,
               use_speaker_embedding=False,
               path_to_checkpoint=None,
               fine_tune=False,
               collapse_margin=0.3):
    """
    Args:
        collapse_margin: margin in which the loss may increase in one epoch without triggering the soft-reset
        steps: How many steps to train
        lr: The initial learning rate for the optimiser
        path_to_checkpoint: reloads a checkpoint to continue training from there
        fine_tune: whether to load everything from a checkpoint, or only the model parameters
        lang: language of the synthesis
        use_speaker_embedding: whether to expect speaker embeddings
        net: Model to train
        train_dataset: Pytorch Dataset Object for train data
        device: Device to put the loaded tensors on
        save_directory: Where to save the checkpoints
        batch_size: How many elements should be loaded at once
        epochs_per_save: how many epochs to train in between checkpoints
    """
    net = net.to(device)
    scaler = GradScaler()
    previous_error = 999999  # tacotron can collapse sometimes and requires soft-resets. This is to detect collapses.
    train_loader = DataLoader(batch_size=batch_size,
                              dataset=train_dataset,
                              drop_last=True,
                              num_workers=10,
                              pin_memory=True,
                              shuffle=True,
                              prefetch_factor=10,
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
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    if path_to_checkpoint is not None:
        # careful when restarting, plotting data will be overwritten!
        check_dict = torch.load(os.path.join(path_to_checkpoint), map_location=device)
        net.load_state_dict(check_dict["model"])
        if not fine_tune:
            optimizer.load_state_dict(check_dict["optimizer"])
            scaler.load_state_dict(check_dict["scaler"])
            step_counter = check_dict["step_counter"]
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
                                     step=step_counter)
                else:
                    train_loss = net(text=batch[0].to(device),
                                     text_lengths=batch[1].to(device),
                                     speech=batch[2].to(device),
                                     speech_lengths=batch[3].to(device),
                                     step=step_counter,
                                     speaker_embeddings=batch[4].to(device))
                train_losses_this_epoch.append(train_loss.item())
            optimizer.zero_grad()
            scaler.scale(train_loss).backward()
            del train_loss
            step_counter += 1
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=False)
            scaler.step(optimizer)
            scaler.update()
        with torch.no_grad():
            net.eval()
            loss_this_epoch = sum(train_losses_this_epoch) / len(train_losses_this_epoch)
            if previous_error + collapse_margin < loss_this_epoch:
                print("Model Collapse detected! \nPrevious Loss: {}\nNew Loss: {}".format(previous_error, loss_this_epoch))
                print("Trying to reset to a stable state ...")
                path_to_checkpoint = get_most_recent_checkpoint(checkpoint_dir=save_directory)
                check_dict = torch.load(path_to_checkpoint, map_location=device)
                net.load_state_dict(check_dict["model"])
                if not fine_tune:
                    optimizer.load_state_dict(check_dict["optimizer"])
                    step_counter = check_dict["step_counter"]
                    scaler.load_state_dict(check_dict["scaler"])
            else:
                previous_error = loss_this_epoch
                if epoch % epochs_per_save == 0:
                    torch.save({
                        "model": net.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "step_counter": step_counter,
                    }, os.path.join(save_directory, "checkpoint_{}.pt".format(step_counter)))
                    delete_old_checkpoints(save_directory, keep=5)
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
                print("Train Loss:   {}".format(loss_this_epoch))
                print("Time elapsed: {} Minutes".format(round((time.time() - start_time) / 60)))
                print("Steps:        {}".format(step_counter))
            torch.cuda.empty_cache()
            net.train()
