import os
import time

import librosa.display as lbd
import matplotlib.pyplot as plt
import torch
import torch.multiprocessing
import torch.multiprocessing
import wandb
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
from Preprocessing.TextFrontend import get_language_id
from TrainingInterfaces.Spectrogram_to_Embedding.StyleEmbedding import StyleEmbedding
from Utility.WarmupScheduler import WarmupScheduler
from Utility.utils import cumsum_durations
from Utility.utils import delete_old_checkpoints
from Utility.utils import get_most_recent_checkpoint
from Utility.storage_config import MODELS_DIR


@torch.no_grad()
def plot_progress_spec(net, device, save_dir, step, lang, default_emb):
    tf = ArticulatoryCombinedTextFrontend(language=lang)
    sentence = ""
    if lang == "en":
        sentence = "This is a complex sentence, it even has a pause!"
    elif lang == "de":
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
    phoneme_vector = tf.string_to_tensor(sentence).squeeze(0).to(device)
    spec, durations, pitch, energy = net.inference(text=phoneme_vector,
                                                   return_duration_pitch_energy=True,
                                                   utterance_embedding=default_emb,
                                                   lang_id=get_language_id(lang).to(device))
    spec = spec.transpose(0, 1).to("cpu").numpy()
    duration_splits, label_positions = cumsum_durations(durations.cpu().numpy())
    if not os.path.exists(os.path.join(save_dir, "spec")):
        os.makedirs(os.path.join(save_dir, "spec"))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
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
    phones = tf.get_phone_string(sentence, for_plot_labels=True)
    ax.set_xticklabels(phones)
    word_boundaries = list()
    for label_index, word_boundary in enumerate(phones):
        if word_boundary == "|":
            word_boundaries.append(label_positions[label_index])
    ax.vlines(x=duration_splits, colors="green", linestyles="dotted", ymin=0.0, ymax=8000, linewidth=1.0)
    ax.vlines(x=word_boundaries, colors="orange", linestyles="dotted", ymin=0.0, ymax=8000, linewidth=1.0)
    pitch_array = pitch.cpu().numpy()
    for pitch_index, xrange in enumerate(zip(duration_splits[:-1], duration_splits[1:])):
        if pitch_array[pitch_index] > 0.001:
            ax.hlines(pitch_array[pitch_index] * 1000, xmin=xrange[0], xmax=xrange[1], color="blue", linestyles="solid",
                      linewidth=0.5)
    ax.set_title(sentence)
    plt.savefig(os.path.join(os.path.join(save_dir, "spec"), str(step) + ".png"))
    plt.clf()
    plt.close()
    return os.path.join(os.path.join(save_dir, "spec"), str(step) + ".png")


def collate_and_pad(batch):
    # text, text_len, speech, speech_len, durations, energy, pitch, utterance condition, language_id
    return (pad_sequence([datapoint[0] for datapoint in batch], batch_first=True),
            torch.stack([datapoint[1] for datapoint in batch]).squeeze(1),
            pad_sequence([datapoint[2] for datapoint in batch], batch_first=True),
            torch.stack([datapoint[3] for datapoint in batch]).squeeze(1),
            pad_sequence([datapoint[4] for datapoint in batch], batch_first=True),
            pad_sequence([datapoint[5] for datapoint in batch], batch_first=True),
            pad_sequence([datapoint[6] for datapoint in batch], batch_first=True),
            None,
            torch.stack([datapoint[8] for datapoint in batch]))


def train_loop(net,
               train_dataset,
               device,
               save_directory,
               batch_size=32,
               epochs_per_save=1,
               lang="en",
               lr=0.0001,
               warmup_steps=4000,
               path_to_checkpoint=None,
               path_to_embed_model=os.path.join(MODELS_DIR, "Embedding", "embedding_function.pt"),
               fine_tune=False,
               resume=False,
               phase_1_steps=100000,
               phase_2_steps=100000,
               use_wandb=False):
    """
    Args:
        resume: whether to resume from the most recent checkpoint
        warmup_steps: how long the learning rate should increase before it reaches the specified value
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
        phase_1_steps: how many steps to train before using any of the cycle objectives
        phase_2_steps: how many steps to train using the cycle objectives
        path_to_embed_model: path to the pretrained embedding function
    """

    steps = phase_1_steps + phase_2_steps

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
                              num_workers=8,
                              pin_memory=True,
                              shuffle=True,
                              prefetch_factor=8,
                              collate_fn=collate_and_pad,
                              persistent_workers=True)
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
        cycle_losses_this_epoch = list()
        for batch in tqdm(train_loader):
            with autocast():
                if step_counter <= phase_1_steps:
                    # ===============================================
                    # =        PHASE 1: no cycle objective          =
                    # ===============================================
                    style_embedding = style_embedding_function(batch_of_spectrograms=batch[2].to(device),
                                                               batch_of_spectrogram_lengths=batch[3].to(device))

                    train_loss = net(text_tensors=batch[0].to(device),
                                     text_lengths=batch[1].to(device),
                                     gold_speech=batch[2].to(device),
                                     speech_lengths=batch[3].to(device),
                                     gold_durations=batch[4].to(device),
                                     gold_pitch=batch[6].to(device),  # mind the switched order
                                     gold_energy=batch[5].to(device),  # mind the switched order
                                     utterance_embedding=style_embedding,
                                     lang_ids=batch[8].to(device),
                                     return_mels=False)
                    train_losses_this_epoch.append(train_loss.item())

                else:
                    # ================================================
                    # = PHASE 2:     cycle objective is added        =
                    # ================================================
                    style_embedding_function.eval()
                    style_embedding_of_gold, out_list_gold = style_embedding_function(
                        batch_of_spectrograms=batch[2].to(device),
                        batch_of_spectrogram_lengths=batch[3].to(device),
                        return_all_outs=True)

                    train_loss, output_spectrograms = net(text_tensors=batch[0].to(device),
                                                          text_lengths=batch[1].to(device),
                                                          gold_speech=batch[2].to(device),
                                                          speech_lengths=batch[3].to(device),
                                                          gold_durations=batch[4].to(device),
                                                          gold_pitch=batch[6].to(device),  # mind the switched order
                                                          gold_energy=batch[5].to(device),  # mind the switched order
                                                          utterance_embedding=style_embedding_of_gold.detach(),
                                                          lang_ids=batch[8].to(device),
                                                          return_mels=True)
                    style_embedding_function.train()
                    style_embedding_of_predicted, out_list_predicted = style_embedding_function(
                        batch_of_spectrograms=output_spectrograms,
                        batch_of_spectrogram_lengths=batch[3].to(device),
                        return_all_outs=True)

                    cycle_dist = 0
                    for out_gold, out_pred in zip(out_list_gold, out_list_predicted):
                        # essentially feature matching, as is often done in vocoder training,
                        # since we're essentially dealing with a discriminator here.
                        cycle_dist = cycle_dist + torch.nn.functional.l1_loss(out_pred, out_gold.detach())

                    train_losses_this_epoch.append(train_loss.item())
                    cycle_losses_this_epoch.append(cycle_dist.item())
                    train_loss = train_loss + cycle_dist

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
            default_embedding = style_embedding_function(
                batch_of_spectrograms=train_dataset[0][2].unsqueeze(0).to(device),
                batch_of_spectrogram_lengths=train_dataset[0][3].unsqueeze(0).to(device)).squeeze()
            torch.save({
                "model":        net.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "step_counter": step_counter,
                "scaler":       scaler.state_dict(),
                "scheduler":    scheduler.state_dict(),
                "default_emb":  default_embedding,
            }, os.path.join(save_directory, "checkpoint_{}.pt".format(step_counter)))
            delete_old_checkpoints(save_directory, keep=5)
            path_to_most_recent_plot = plot_progress_spec(net,
                                                          device,
                                                          save_dir=save_directory,
                                                          step=step_counter,
                                                          lang=lang,
                                                          default_emb=default_embedding)
            if use_wandb:
                wandb.log({
                    "progress_plot": wandb.Image(path_to_most_recent_plot)
                })
        print("Epoch:              {}".format(epoch))
        print("Spectrogram Loss:   {}".format(sum(train_losses_this_epoch) / len(train_losses_this_epoch)))
        if len(cycle_losses_this_epoch) != 0:
            print("Cycle Loss:         {}".format(sum(cycle_losses_this_epoch) / len(cycle_losses_this_epoch)))
        print("Time elapsed:       {} Minutes".format(round((time.time() - start_time) / 60)))
        print("Steps:              {}".format(step_counter))
        if use_wandb:
            wandb.log({
                "spectrogram_loss": sum(train_losses_this_epoch) / len(train_losses_this_epoch),
                "cycle_loss":       sum(cycle_losses_this_epoch) / len(cycle_losses_this_epoch) if len(
                    cycle_losses_this_epoch) != 0 else 0.0,
                "epoch":            epoch,
                "steps":            step_counter,
            })
        if step_counter > steps and epoch % epochs_per_save == 0:
            # DONE
            return
        net.train()
