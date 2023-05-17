import os
import time
import random

import torch
import torch.multiprocessing
import wandb
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from TrainingInterfaces.Spectrogram_to_Embedding.StyleEmbedding import StyleEmbedding
from Utility.WarmupScheduler import ToucanWarmupScheduler as WarmupScheduler

from Utility.utils import delete_old_checkpoints
from Utility.utils import get_most_recent_checkpoint

def collate_and_pad(batch):
    # speech, speech_len, sentence string, filepaths
    return (pad_sequence([datapoint[2] for datapoint in batch], batch_first=True),
            torch.stack([datapoint[3] for datapoint in batch]).squeeze(1),
            [datapoint[9] for datapoint in batch],
            [datapoint[10] for datapoint in batch])

def train_loop(net,
               train_dataset,
               device,
               save_directory,
               batch_size,
               lr,
               warmup_steps,
               path_to_checkpoint,
               path_to_embed_model,
               fine_tune,
               resume,
               steps,
               use_wandb,
               sent_embs=None,
               random_emb=False,
               emovdb=False):
    net = net.to(device)

    style_embedding_function = StyleEmbedding().to(device)
    check_dict = torch.load(path_to_embed_model, map_location=device)
    style_embedding_function.load_state_dict(check_dict["style_emb_func"])
    style_embedding_function.eval()
    style_embedding_function.requires_grad_(False)

    torch.multiprocessing.set_sharing_strategy('file_system')
    train_loader = DataLoader(batch_size=batch_size,
                              dataset=train_dataset,
                              drop_last=True,
                              num_workers=12 if os.cpu_count() > 12 else max(os.cpu_count() - 2, 1),
                              pin_memory=True,
                              shuffle=True,
                              prefetch_factor=2,
                              collate_fn=collate_and_pad,
                              persistent_workers=True)
    step_counter = 0
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = WarmupScheduler(optimizer, peak_lr=lr, warmup_steps=warmup_steps, max_steps=steps)
    epoch = 0
    if resume:
        path_to_checkpoint = get_most_recent_checkpoint(checkpoint_dir=save_directory)
    if path_to_checkpoint is not None:
        check_dict = torch.load(path_to_checkpoint, map_location=device)
        net.load_state_dict(check_dict["model"])
        if not fine_tune:
            optimizer.load_state_dict(check_dict["optimizer"])
            scheduler.load_state_dict(check_dict["scheduler"])
            step_counter = check_dict["step_counter"]
    start_time = time.time()
    while True:
        net.train()
        epoch += 1
        sent_style_losses_total = list()

        for batch in tqdm(train_loader):
            train_loss = 0.0
            style_embedding = style_embedding_function(batch_of_spectrograms=batch[0].to(device),
                                                       batch_of_spectrogram_lengths=batch[1].to(device))
            if emovdb:
                filepaths = batch[3]
                if random_emb:
                    emotions = [os.path.splitext(os.path.basename(path))[0].split("-16bit")[0].split("_")[0].lower() for path in filepaths]
                    sentence_embedding = torch.stack([random.choice(sent_embs[emotion]) for emotion in emotions]).to(device)
                else:
                    sentence_embedding = torch.stack([sent_embs[path] for path in filepaths]).to(device)
            else:
                sentences = batch[2]
                sentence_embedding = torch.stack([sent_embs[sent] for sent in sentences]).to(device)

            sent_style_loss = net(style_embedding=style_embedding,
                                  sentence_embedding=sentence_embedding)
            if sent_style_loss is not None:
                if not torch.isnan(sent_style_loss):
                    train_loss = train_loss + sent_style_loss
                sent_style_losses_total.append(sent_style_loss.item())
            
            optimizer.zero_grad()
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=False)
            optimizer.step()
            scheduler.step()
            step_counter += 1

        # EPOCH IS OVER
        net.eval()
        style_embedding_function.eval()
        torch.save({
            "model"       : net.state_dict(),
            "optimizer"   : optimizer.state_dict(),
            "step_counter": step_counter,
            "scheduler"   : scheduler.state_dict(),
        }, os.path.join(save_directory, "checkpoint_{}.pt".format(step_counter)))
        delete_old_checkpoints(save_directory, keep=5)

        print("\nEpoch:                  {}".format(epoch))
        print("Time elapsed:           {} Minutes".format(round((time.time() - start_time) / 60)))
        print("Sentence Style Loss:    {}".format(round(sum(sent_style_losses_total) / len(sent_style_losses_total), 5)))
        print("Steps:                  {}\n".format(step_counter))
        if use_wandb:
            wandb.log({
                "sentence_style_loss": round(sum(sent_style_losses_total) / len(sent_style_losses_total), 5),
            }, step=step_counter)
        
        if step_counter > steps:
            return  # DONE
        
        net.train()
