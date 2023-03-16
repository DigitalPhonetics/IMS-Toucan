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
import numpy as np
from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
from Preprocessing.TextFrontend import get_language_id
from Utility.WarmupScheduler import WarmupScheduler
from Utility.utils import cumsum_durations
from Utility.utils import delete_old_checkpoints
from Utility.utils import get_most_recent_checkpoint
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.logger import FastSpeech2Logger #Added for logging
from Preprocessing.Language_embedding import LanguageEmbedding

#This function creates the logging directory and the returns the logger.
def prepare_logger(log_directory):
    if not os.path.isdir(log_directory):
        os.makedirs(log_directory)
        os.chmod(log_directory, 0o775)
    logger = FastSpeech2Logger( log_directory)
    return logger

@torch.no_grad()
def plot_progress_spec(net, device, save_dir, step, lang, default_emb):
    tf = ArticulatoryCombinedTextFrontend(language=lang)
    sentence = ""
    if lang == "en":
        sentence = "This is a complex sentence, it even has a pause!"
    elif lang == "de":
        sentence = "Dies ist ein komplexer Satz, er hat sogar eine Pause!"
    elif lang == "at":
        sentence = "Dies ist ein komplexer Satz, er hat sogar eine Pause!"
    elif lang == "vd":
        sentence = "Dies ist ein komplexer Satz, er hat sogar eine Pause!"
    elif lang == "at-lab":
        sentence = "Dies ist ein komplexer Satz, er hat sogar eine Pause!"
    elif lang == "el":
        sentence = "Αυτή είναι μια σύνθετη πρόταση, έχει ακόμη και παύση!"
    elif lang == "es":
        sentence = "Esta es una oración compleja, ¡incluso tiene una pausa!"
    elif lang == "fi":
        sentence = "Tämä on monimutkainen lause, sillä on jopa tauko!"
    elif lang == "ru":
        sentence = "Это сложное предложение, в нем даже есть пауза!"
    elif lang == "hu":
        sentence = "Ez egy összetett mondat, még szünet is van benne!"
    elif lang == "nl":
        sentence = "Dit is een complexe zin, er zit zelfs een pauze in!"
    elif lang == "fr":
        sentence = "C'est une phrase complexe, elle a même une pause !"
    elif lang == "pt":
        sentence = "Esta é uma frase complexa, tem até uma pausa!"
    elif lang == "pl":
        sentence = "To jest zdanie złożone, ma nawet pauzę!"
    elif lang == "it":
        sentence = "Questa è una frase complessa, ha anche una pausa!"
    elif lang == "cmn":
        sentence = "这是一个复杂的句子，它甚至包含一个停顿。"
    elif lang == "vi":
        sentence = "Đây là một câu phức tạp, nó thậm chí còn chứa một khoảng dừng."
    if lang == "at-lab":
        sentence == "Heute ist schönes Frühlingswetter!"
        phoneme_vector = tf.string_to_tensor(sentence, path_to_wavfile="/data/vokquant/data/aridialect/aridialect_wav16000/alf_at_berlin_001.wav").squeeze(0).to(device)
    else:
        phoneme_vector = tf.string_to_tensor(sentence).squeeze(0).to(device)
    emb = LanguageEmbedding()
    spec, durations, *_ = net.inference(text=phoneme_vector,
                                        return_duration_pitch_energy=True,
                                        utterance_embedding=default_emb,
                                        lang_emb=emb.get_emb_from_path(path_to_wavfile="/data/vokquant/data/aridialect/aridialect_wav16000/alf_at_berlin_001.wav" ).to(device))
    spec = spec.transpose(0, 1).to("cpu").numpy()
    duration_splits, label_positions = cumsum_durations(durations.cpu().numpy())
    if not os.path.exists(os.path.join(save_dir, "spec")):
        os.makedirs(os.path.join(save_dir, "spec"))
    fig, ax = plt.subplots(nrows=1, ncols=1)
    lbd.specshow(spec,
                 ax=ax,
                 sr=16000,
                 cmap='GnBu',
                 y_axis='mel',
                 x_axis=None,
                 hop_length=256)
    ax.yaxis.set_visible(False)
    ax.set_xticks(duration_splits, minor=True)
    ax.xaxis.grid(True, which='minor')
    ax.set_xticks(label_positions, minor=False)
    ax.set_xticklabels(tf.get_phone_string(sentence, for_plot_labels=True, path_to_wavfile="/data/vokquant/data/aridialect/aridialect_wav16000/alf_at_berlin_001.wav"))
    ax.set_title(sentence)
    plt.savefig(os.path.join(os.path.join(save_dir, "spec"), str(step) + ".png"))
    plt.clf()
    plt.close()


def collate_and_pad(batch):
    # text, text_len, speech, speech_len, durations, energy, pitch, utterance condition,language_embedding, language_id
    if type([datapoint[8] for datapoint in batch][0]) == np.ndarray:
        lang_emb = [torch.from_numpy(datapoint[8]) for datapoint in batch]
    else:
        lang_emb = [datapoint[8] for datapoint in batch]
    return (pad_sequence([datapoint[0] for datapoint in batch], batch_first=True),
            torch.stack([datapoint[1] for datapoint in batch]).squeeze(1),
            pad_sequence([datapoint[2] for datapoint in batch], batch_first=True),
            torch.stack([datapoint[3] for datapoint in batch]).squeeze(1),
            pad_sequence([datapoint[4] for datapoint in batch], batch_first=True),
            pad_sequence([datapoint[5] for datapoint in batch], batch_first=True),
            pad_sequence([datapoint[6] for datapoint in batch], batch_first=True),
            torch.stack([datapoint[7] for datapoint in batch]).squeeze(),
            torch.stack(lang_emb).squeeze(),
            torch.stack([datapoint[9] for datapoint in batch]))


def train_loop(net,
               train_dataset,
               device,
               save_directory,
               batch_size=32,
               steps=300000,
               epochs_per_save=1,
               lang="en",
               lr=0.0001,
               warmup_steps=4000,
               path_to_checkpoint=None,
               fine_tune=False,
               resume=False):
    """
    Args:
        resume: whether to resume from the most recent checkpoint
        warmup_steps: how long the learning rate should increase before it reaches the specified value
        steps: How many steps to train
        lr: The initial learning rate for the optimiser
        path_to_checkpoint: reloads a checkpoint to continue training from there
        fine_tune: whether to load everything from a checkpoint, or only the model parameters
        lang: language of the synthesis
        net: Model to train
        train_dataset: Pytorch Dataset Object for train data
        device: Device to put the loaded tensors on
        save_directory: Where to save the checkpoints
        batch_size: How many elements should be loaded at once
        epochs_per_save: how many epochs to train in between checkpoints

    """
    net = net.to(device)

    logger = prepare_logger(log_directory=os.path.join(save_directory,'logs')) #Create the logger and the logging dir

    torch.multiprocessing.set_sharing_strategy('file_system')
    train_loader = DataLoader(batch_size=batch_size,
                              dataset=train_dataset,
                              drop_last=True,
                              num_workers=8,
                              pin_memory=True,
                              shuffle=True,
                              prefetch_factor=8,
                              collate_fn=collate_and_pad,
                              persistent_workers=True)
    default_embedding = None
    for index in range(20):  # slicing is not implemented for datasets, so this detour is needed.
        if default_embedding is None:
            default_embedding = train_dataset[index][7].squeeze()
        else:
            default_embedding = default_embedding + train_dataset[index][7].squeeze()
    default_embedding = (default_embedding / len(train_dataset)).to(device)
    # default speaker embedding for inference is the average of the first 20 speaker embeddings. So if you use multiple datasets combined,
    # put a single speaker one with the nicest voice first into the concat dataset.
    step_counter = 0
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = WarmupScheduler(optimizer, warmup_steps=warmup_steps)
    scaler = GradScaler()
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
            scaler.load_state_dict(check_dict["scaler"])
    start_time = time.time()
    while True:
        net.train()
        epoch += 1
        optimizer.zero_grad()
        train_losses_this_epoch = list()
        for batch in tqdm(train_loader):
            with autocast():
                train_loss = net(text_tensors=batch[0].to(device),
                                 text_lengths=batch[1].to(device),
                                 gold_speech=batch[2].to(device),
                                 speech_lengths=batch[3].to(device),
                                 gold_durations=batch[4].to(device),
                                 gold_pitch=batch[6].to(device),  # mind the switched order
                                 gold_energy=batch[5].to(device),  # mind the switched order
                                 utterance_embedding=batch[7].to(device),
                                 lang_embs=batch[8].to(device),
                                 return_mels=False)
                train_losses_this_epoch.append(train_loss.item())

            optimizer.zero_grad()
            scaler.scale(train_loss).backward()
            del train_loss
            step_counter += 1
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=False)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        net.eval()
        if epoch % epochs_per_save == 0:
            torch.save({
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step_counter": step_counter,
                "scaler": scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
                "default_emb": default_embedding,
            }, os.path.join(save_directory, "checkpoint_{}.pt".format(step_counter)))
            delete_old_checkpoints(save_directory, keep=5)
            plot_progress_spec(net, device, save_dir=save_directory, step=step_counter, lang=lang, default_emb=default_embedding)
            if step_counter > steps:
                # DONE
                return
        print("Epoch:        {}".format(epoch))
        print("Train Loss:   {}".format(sum(train_losses_this_epoch) / len(train_losses_this_epoch)))
        print("Time elapsed: {} Minutes".format(round((time.time() - start_time) / 60)))
        print("Steps:        {}".format(step_counter))

        logger.log_training(sum(train_losses_this_epoch) / len(train_losses_this_epoch),step_counter) #We add the loss of the specific step to the log

        net.train()
