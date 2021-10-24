import os
import time

import matplotlib.pyplot as plt
import torch
import torch.multiprocessing
import torch.nn.functional as F
from speechbrain.pretrained import EncoderClassifier
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from Preprocessing.ArticulatoryCombinedTextFrontend import ArticulatoryCombinedTextFrontend
from Utility.utils import delete_old_checkpoints
from Utility.utils import get_most_recent_checkpoint


def plot_attention(model, lang, device, att_dir, step):
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
    text = tf.string_to_tensor(sentence).to(device)
    phones = tf.get_phone_string(sentence)
    model.eval()
    _, _, att, align_att = model.inference(text_tensor=text, return_align_att=True)
    att, align_att = att.to("cpu").detach(), align_att.to("cpu").detach()
    model.train()
    del tf
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 9))
    ax[0].imshow(att.numpy(), interpolation='nearest', aspect='auto', origin="lower")
    ax[1].imshow(align_att, interpolation='nearest', aspect='auto', origin="lower")
    ax[1].set_xlabel("Inputs")
    ax[0].xaxis.set_visible(False)
    ax[0].set_ylabel("Outputs")
    ax[1].set_ylabel("Outputs")
    ax[1].set_xticks(range(len(att[0])))
    del att
    ax[1].set_xticklabels(labels=[phone for phone in phones])
    ax[0].set_title("Encoder-Decoder-Attention")
    ax[1].set_title("Alignment-Attention")
    fig.tight_layout()
    if not os.path.exists(os.path.join(att_dir, "attention_plots")):
        os.makedirs(os.path.join(att_dir, "attention_plots"))
    fig.savefig(os.path.join(os.path.join(att_dir, "attention_plots"), str(step) + ".png"))
    fig.clf()
    plt.close()


def collate_and_pad(batch):
    # text, text_len, speech, speech_len
    return (pad_sequence([datapoint[0] for datapoint in batch], batch_first=True),
            torch.stack([datapoint[1] for datapoint in batch]).squeeze(1),
            pad_sequence([datapoint[2] for datapoint in batch], batch_first=True),
            torch.stack([datapoint[3] for datapoint in batch]).squeeze(1))


def train_loop(net,
               train_dataset,
               device,
               save_directory,
               batch_size,
               steps,
               epochs_per_save,
               lang,
               lr,
               path_to_checkpoint=None,
               fine_tune=False,
               collapse_margin=5.0,  # be wary of loss scheduling
               resume=False,
               cycle_loss_start_steps=None,
               silent=False):
    """
    Args:
        silent: whether to print things, which can be problematic in multiprocessing when all processes print over each other
        cycle_loss_start_steps: after how many steps the cycle consistency loss for voice identity should start
        resume: whether to resume from the most recent checkpoint
        collapse_margin: margin in which the loss may increase in one epoch without triggering the soft-reset
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
    previous_error = 999999  # tacotron can collapse sometimes and requires soft-resets. This is to detect collapses.
    train_loader = DataLoader(batch_size=batch_size,
                              dataset=train_dataset,
                              drop_last=True,
                              num_workers=8,
                              pin_memory=False,
                              shuffle=True,
                              prefetch_factor=16,
                              collate_fn=collate_and_pad,
                              persistent_workers=True)

    if cycle_loss_start_steps is not None:
        speaker_embedding_func = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                                run_opts={"device": str(device)},
                                                                savedir="Models/SpeakerEmbedding/speechbrain_speaker_embedding_ecapa")
    else:
        speaker_embedding_func = None
        cycle_loss_start_steps = 0
    step_counter = 0
    epoch = 0
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1.0e-06, weight_decay=0.0)
    scaler = GradScaler()
    if resume:
        previous_checkpoint = get_most_recent_checkpoint(checkpoint_dir=save_directory)
        if previous_checkpoint is not None:
            path_to_checkpoint = previous_checkpoint
            fine_tune = False
        else:
            fine_tune = True

    if path_to_checkpoint is not None:
        check_dict = torch.load(os.path.join(path_to_checkpoint), map_location=device)
        net.load_state_dict(check_dict["model"])
        if not fine_tune:
            optimizer.load_state_dict(check_dict["optimizer"])
            step_counter = check_dict["step_counter"]
            scaler.load_state_dict(check_dict["scaler"])
            if step_counter > steps:
                print("Desired steps already reached in loaded checkpoint.")
                return
    start_time = time.time()

    while True:
        cumulative_loss_dict = dict()
        epoch += 1
        net.train()
        optimizer.zero_grad()
        train_losses_this_epoch = list()
        for batch in tqdm(train_loader):
            with autocast():
                train_loss, predicted_mels, loss_dict = net(text=batch[0].to(device),
                                                            text_lengths=batch[1].to(device),
                                                            speech=batch[2].to(device),
                                                            speech_lengths=batch[3].to(device),
                                                            step=step_counter,
                                                            return_mels=True,
                                                            return_loss_dict=True)

                if step_counter > cycle_loss_start_steps and speaker_embedding_func is not None:
                    pred_spemb = speaker_embedding_func.modules.embedding_model(predicted_mels,
                                                                                torch.tensor([x / len(predicted_mels[0]) for x in batch[3]]))
                    gold_spemb = speaker_embedding_func.modules.embedding_model(batch[2].to(device),
                                                                                torch.tensor([x / len(batch[2][0]) for x in batch[3]]))
                    # we have to calculate the speaker embedding from our own melspec because we project into a slightly different melspec space
                    cosine_cycle_distance = torch.tensor(1.0) - F.cosine_similarity(pred_spemb.squeeze(), gold_spemb.squeeze(), dim=1).mean()
                    pairwise_cycle_distance = F.pairwise_distance(pred_spemb.squeeze(), gold_spemb.squeeze()).mean()
                    cycle_distance = cosine_cycle_distance + pairwise_cycle_distance
                    del pred_spemb
                    del predicted_mels
                    del gold_spemb
                    cycle_loss = cycle_distance * min(1.0, (step_counter - cycle_loss_start_steps) / 100000)
                    loss_dict["cycle"] = cycle_loss.item()
                    train_loss = train_loss + cycle_loss

                train_losses_this_epoch.append(train_loss.item())
                for loss_type in loss_dict:
                    if loss_type not in cumulative_loss_dict.keys():
                        cumulative_loss_dict[loss_type] = list()
                    cumulative_loss_dict[loss_type].append(loss_dict[loss_type])

            optimizer.zero_grad()
            if speaker_embedding_func is not None:
                speaker_embedding_func.modules.embedding_model.zero_grad()
            scaler.scale(train_loss).backward()
            del train_loss
            step_counter += 1
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=False)
            scaler.step(optimizer)
            scaler.update()
        net.eval()
        loss_this_epoch = sum(train_losses_this_epoch) / len(train_losses_this_epoch)
        if previous_error + collapse_margin < loss_this_epoch:
            print(f"Model Collapse detected in {lang}! \nPrevious Loss: {previous_error}\nNew Loss: {loss_this_epoch}")
            print("Trying to reset to a stable state ...")
            path_to_checkpoint = get_most_recent_checkpoint(checkpoint_dir=save_directory)
            check_dict = torch.load(path_to_checkpoint, map_location=device)
            net.load_state_dict(check_dict["model"])
            # the rest is assuming that there was at least one successful checkpoint, otherwise the finetuning flag is lost
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
                },
                    os.path.join(save_directory, "checkpoint_{}.pt".format(step_counter)))
                delete_old_checkpoints(save_directory, keep=5)
                with torch.no_grad():
                    plot_attention(model=net,
                                   lang=lang,
                                   device=device,
                                   att_dir=save_directory,
                                   step=step_counter)
                if step_counter > steps:
                    if not silent:
                        print("Epoch:        {}".format(epoch))
                        print("Total Loss:   {}".format(round(loss_this_epoch, 3)))
                        for loss_type in cumulative_loss_dict:
                            print(f"    {loss_type}: {round(sum(cumulative_loss_dict[loss_type]) / len(cumulative_loss_dict[loss_type]), 3)}")
                        print("Time elapsed: {} Minutes".format(round((time.time() - start_time) / 60)))
                        print("Steps:        {}".format(step_counter))
                    # DONE
                    return
            if not silent:
                print("Epoch:        {}".format(epoch))
                print("Total Loss:   {}".format(round(loss_this_epoch, 3)))
                for loss_type in cumulative_loss_dict:
                    print(f"    {loss_type}: {round(sum(cumulative_loss_dict[loss_type]) / len(cumulative_loss_dict[loss_type]), 3)}")
                print("Time elapsed: {} Minutes".format(round((time.time() - start_time) / 60)))
                print("Steps:        {}".format(step_counter))
        net.train()
