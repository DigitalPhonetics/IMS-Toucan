import os
import time

import librosa.display as lbd
import matplotlib.pyplot as plt
import torch
import torch.multiprocessing
import torch.multiprocessing
import torch.nn.functional as F
from speechbrain.pretrained import EncoderClassifier
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from Preprocessing.ArticulatoryCombinedTextFrontend import ArticulatoryCombinedTextFrontend
from Utility.WarmupScheduler import WarmupScheduler
from Utility.utils import cumsum_durations
from Utility.utils import delete_old_checkpoints
from Utility.utils import get_most_recent_checkpoint


def plot_progress_spec(net, device, save_dir, step, lang):
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
    phoneme_vector = tf.string_to_tensor(sentence).squeeze(0).to(device)
    spec, durations, *_ = net.inference(text=phoneme_vector, return_duration_pitch_energy=True)
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
    ax.set_xticklabels(tf.get_phone_string(sentence))
    ax.set_title(sentence)
    plt.savefig(os.path.join(os.path.join(save_dir, "spec"), str(step) + ".png"))
    plt.clf()
    plt.close()


def collate_and_pad(batch):
    # text, text_len, speech, speech_len, durations, energy, pitch
    return (pad_sequence([datapoint[0] for datapoint in batch], batch_first=True),
            torch.stack([datapoint[1] for datapoint in batch]).squeeze(1),
            pad_sequence([datapoint[2] for datapoint in batch], batch_first=True),
            torch.stack([datapoint[3] for datapoint in batch]).squeeze(1),
            pad_sequence([datapoint[4] for datapoint in batch], batch_first=True),
            pad_sequence([datapoint[5] for datapoint in batch], batch_first=True),
            pad_sequence([datapoint[6] for datapoint in batch], batch_first=True))


def train_loop(net,
               train_dataset,
               device,
               save_directory,
               batch_size=32,
               steps=300000,
               epochs_per_save=5,
               lang="en",
               lr=0.0001,
               warmup_steps=14000,
               path_to_checkpoint=None,
               fine_tune=False,
               resume=False,
               cycle_loss_start_steps=None):
    """
    Args:
        cycle_loss_start_steps: after how many steps the cycle consistency loss for voice identity should start
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
    if cycle_loss_start_steps is not None:
        speaker_embedding_func = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                                run_opts={"device": str(device)},
                                                                savedir="Models/SpeakerEmbedding/speechbrain_speaker_embedding_ecapa")
    else:
        speaker_embedding_func = None
        cycle_loss_start_steps = 0

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
                train_loss, predicted_mels = net(text_tensors=batch[0].to(device),
                                                 text_lengths=batch[1].to(device),
                                                 gold_speech=batch[2].to(device),
                                                 speech_lengths=batch[3].to(device),
                                                 gold_durations=batch[4].to(device),
                                                 gold_pitch=batch[6].to(device),  # mind the change of order here!
                                                 gold_energy=batch[5].to(device),
                                                 return_mels=True)

                train_losses_this_epoch.append(train_loss.item())
                if step_counter > cycle_loss_start_steps and speaker_embedding_func is not None:
                    pred_spemb = speaker_embedding_func.modules.embedding_model(predicted_mels,
                                                                                torch.tensor([x / len(predicted_mels[0]) for x in batch[3]]))
                    gold_spemb = speaker_embedding_func.modules.embedding_model(batch[2].to(device),
                                                                                torch.tensor([x / len(batch[2][0]) for x in batch[3]]))
                    # we have to recalculate the speaker embedding from our own mel because we project into a slightly different mel space
                    cycle_distance = torch.tensor(1.0) - F.cosine_similarity(pred_spemb.squeeze(), gold_spemb.squeeze(), dim=1).mean()
                    del pred_spemb
                    del predicted_mels
                    del gold_spemb
                    cycle_loss = cycle_distance * min(1.0, (step_counter - cycle_loss_start_steps) / 500)
                    train_loss = train_loss + cycle_loss

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
            scheduler.step()

        net.eval()
        if epoch % epochs_per_save == 0:
            torch.save({
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step_counter": step_counter,
                "scaler": scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, os.path.join(save_directory, "checkpoint_{}.pt".format(step_counter)))
            delete_old_checkpoints(save_directory, keep=5)
            with torch.no_grad():
                plot_progress_spec(net, device, save_dir=save_directory, step=step_counter, lang=lang)
            if step_counter > steps:
                # DONE
                return
        print("Epoch:        {}".format(epoch))
        print("Train Loss:   {}".format(sum(train_losses_this_epoch) / len(train_losses_this_epoch)))
        print("Time elapsed: {} Minutes".format(round((time.time() - start_time) / 60)))
        print("Steps:        {}".format(step_counter))
        net.train()
