import os
import random
import time

import librosa.display as lbd
import matplotlib.pyplot as plt
import torch
import torch.multiprocessing
import torch.multiprocessing
from torch.cuda.amp import GradScaler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from Preprocessing.ArticulatoryCombinedTextFrontend import ArticulatoryCombinedTextFrontend
from TrainingInterfaces.Text_to_Spectrogram.AutoAligner.Aligner import Aligner
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


def train_loop(net,
               train_sentences,
               device,
               save_directory,
               aligner_checkpoint,
               batch_size=32,
               steps=300000,
               epochs_per_save=5,
               lang="en",
               lr=0.0001,
               warmup_steps=14000,
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
        lang: language of the synthesis and of the train sentences
        net: Model to train
        train_sentences: list of (string) sentences the CTC objective should be learned on
        device: Device to put the loaded tensors on
        save_directory: Where to save the checkpoints
        batch_size: How many elements should be loaded at once
        epochs_per_save: how many epochs to train in between checkpoints

    """
    net = net.to(device)

    torch.multiprocessing.set_sharing_strategy('file_system')
    text_to_art_vec = ArticulatoryCombinedTextFrontend(language=lang)
    asr_aligner = Aligner().to(device)
    check_dict = torch.load(os.path.join(aligner_checkpoint), map_location=device)
    asr_aligner.load_state_dict(check_dict["asr_model"])
    net.stop_gradient_from_energy_predictor = False
    net.stop_gradient_from_pitch_predictor = False
    vector_to_id = dict()
    for phone in text_to_art_vec.phone_to_id:
        vector_to_id[text_to_art_vec.phone_to_vector[phone]] = text_to_art_vec.phone_to_id[phone]
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
        random.shuffle(train_sentences)
        batch_of_text_vecs = list()
        batch_of_tokens = list()

        for sentence in tqdm(train_sentences):
            if sentence.strip() == "":
                continue
            text_vec = text_to_art_vec.string_to_tensor(sentence).squeeze(0).to(device)

            # collect batch of texts
            batch_of_text_vecs.append(text_vec)

            # collect batch of tokens
            tokens = list()
            for vector in text_vec:
                tokens.append(vector_to_id[vector])
            tokens = torch.LongTensor(tokens).to(device)
            batch_of_tokens.append(tokens)

            if len(batch_of_tokens) == batch_size:
                token_batch = pad_sequence(batch_of_tokens, batch_first=True)
                token_lens = torch.LongTensor([len(x) for x in batch_of_tokens]).to(device)
                text_batch = pad_sequence(batch_of_text_vecs, batch_first=True)
                spec_batch, d_outs = net.batch_inference(texts=text_batch, text_lens=token_lens)
                spec_lens = torch.LongTensor([sum(x) for x in d_outs]).to(device)

                asr_pred = asr_aligner(spec_batch, spec_lens)
                train_loss = asr_aligner.ctc_loss(asr_pred.transpose(0, 1).log_softmax(2), token_batch, spec_lens, token_lens)
                train_losses_this_epoch.append(train_loss.item())

                optimizer.zero_grad()
                asr_aligner.zero_grad()
                scaler.scale(train_loss).backward()
                del train_loss
                step_counter += 1
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=False)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                batch_of_tokens = list()
                batch_of_text_vecs = list()

        net.eval()
        if epoch % epochs_per_save == 0:
            torch.save({
                "model"       : net.state_dict(),
                "optimizer"   : optimizer.state_dict(),
                "step_counter": step_counter,
                "scaler"      : scaler.state_dict(),
                "scheduler"   : scheduler.state_dict(),
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
