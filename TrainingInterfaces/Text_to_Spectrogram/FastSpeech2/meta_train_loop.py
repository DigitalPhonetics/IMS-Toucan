import random

import librosa.display as lbd
import matplotlib.pyplot as plt
import torch
import torch.multiprocessing
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
from Preprocessing.TextFrontend import get_language_id
from Utility.WarmupScheduler import WarmupScheduler
from Utility.path_to_transcript_dicts import *
from Utility.utils import cumsum_durations
from Utility.utils import delete_old_checkpoints
from Utility.utils import get_most_recent_checkpoint


def train_loop(net,
               datasets,
               device,
               save_directory,
               batch_size,
               steps,
               steps_per_checkpoint,
               lr,
               path_to_checkpoint,
               resume=False,
               warmup_steps=4000):
    # ============
    # Preparations
    # ============
    net = net.to(device)
    torch.multiprocessing.set_sharing_strategy('file_system')
    train_loaders = list()
    train_iters = list()
    for dataset in datasets:
        train_loaders.append(DataLoader(batch_size=batch_size,
                                        dataset=dataset,
                                        drop_last=True,
                                        num_workers=2,
                                        pin_memory=True,
                                        shuffle=True,
                                        prefetch_factor=5,
                                        collate_fn=collate_and_pad,
                                        persistent_workers=True))
        train_iters.append(iter(train_loaders[-1]))
    languages_used = ["en", "de", "el", "es", "fi", "ru", "hu", "nl", "fr", "pt", "pl", "it", "cmn", "vi"]
    default_embeddings = dict()
    for index, lang in enumerate(languages_used):
        default_embeddings[lang] = datasets[index][0][7].squeeze().to(device)
    optimizer = torch.optim.RAdam(net.parameters(), lr=lr, eps=1.0e-06, weight_decay=0.0)
    grad_scaler = GradScaler()
    scheduler = WarmupScheduler(optimizer, warmup_steps=warmup_steps)
    if resume:
        previous_checkpoint = get_most_recent_checkpoint(checkpoint_dir=save_directory)
        if previous_checkpoint is not None:
            path_to_checkpoint = previous_checkpoint
        else:
            raise RuntimeError(f"No checkpoint found that can be resumed from in {save_directory}")
    step_counter = 0
    train_losses_total = list()
    if path_to_checkpoint is not None:
        check_dict = torch.load(os.path.join(path_to_checkpoint), map_location=device)
        net.load_state_dict(check_dict["model"])
        if resume:
            optimizer.load_state_dict(check_dict["optimizer"])
            step_counter = check_dict["step_counter"]
            grad_scaler.load_state_dict(check_dict["scaler"])
            scheduler.load_state_dict(check_dict["scheduler"])
            if step_counter > steps:
                print("Desired steps already reached in loaded checkpoint.")
                return

    net.train()
    # =============================
    # Actual train loop starts here
    # =============================
    for step in tqdm(range(step_counter, steps)):
        batches = []
        for index in random.sample(list(range(len(datasets))), len(datasets)):
            # we get one batch for each task (i.e. language in this case)
            try:
                batch = next(train_iters[index])
                batches.append(batch)
            except StopIteration:
                train_iters[index] = iter(train_loaders[index])
                batch = next(train_iters[index])
                batches.append(batch)
        train_loss = 0.0
        for batch in batches:
            with autocast():
                # we sum the loss for each task, as we would do for the
                # second order regular MAML, but we do it only over one
                # step (i.e. iterations of inner loop = 1)
                train_loss = train_loss + net(text_tensors=batch[0].to(device),
                                              text_lengths=batch[1].to(device),
                                              gold_speech=batch[2].to(device),
                                              speech_lengths=batch[3].to(device),
                                              gold_durations=batch[4].to(device),
                                              gold_pitch=batch[6].to(device),  # mind the switched order
                                              gold_energy=batch[5].to(device),  # mind the switched order
                                              utterance_embedding=batch[7].to(device),
                                              lang_ids=batch[8].to(device),
                                              return_mels=False)
        # then we directly update our meta-parameters without
        # the need for any task specific parameters
        train_losses_total.append(train_loss.item())
        optimizer.zero_grad()
        grad_scaler.scale(train_loss).backward()
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=False)
        grad_scaler.step(optimizer)
        grad_scaler.update()
        scheduler.step()

        if step % steps_per_checkpoint == 0:
            # ==============================
            # Enough steps for some insights
            # ==============================
            net.eval()
            print(f"Total Loss: {round(sum(train_losses_total) / len(train_losses_total), 3)}")
            train_losses_total = list()
            torch.save({
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": grad_scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step_counter": step,
                "default_emb": default_embeddings["en"]
            },
                os.path.join(save_directory, "checkpoint_{}.pt".format(step)))
            delete_old_checkpoints(save_directory, keep=5)
            for lang in languages_used:
                plot_progress_spec(net=net,
                                   device=device,
                                   lang=lang,
                                   save_dir=save_directory,
                                   step=step,
                                   utt_embeds=default_embeddings)
            net.train()


@torch.inference_mode()
def plot_progress_spec(net, device, save_dir, step, lang, utt_embeds):
    tf = ArticulatoryCombinedTextFrontend(language=lang)
    sentence = ""
    default_embed = utt_embeds[lang]
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
    spec, durations, *_ = net.inference(text=phoneme_vector,
                                        return_duration_pitch_energy=True,
                                        utterance_embedding=default_embed,
                                        lang_id=get_language_id(lang).to(device))
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
    ax.set_xticklabels(tf.get_phone_string(sentence, for_plot_labels=True))
    ax.set_title(sentence)
    plt.savefig(os.path.join(os.path.join(save_dir, "spec"), f"{step}_{lang}.png"))
    plt.clf()
    plt.close()


def collate_and_pad(batch):
    # text, text_len, speech, speech_len, durations, energy, pitch, utterance condition, language_id
    return (pad_sequence([datapoint[0] for datapoint in batch], batch_first=True),
            torch.stack([datapoint[1] for datapoint in batch]).squeeze(1),
            pad_sequence([datapoint[2] for datapoint in batch], batch_first=True),
            torch.stack([datapoint[3] for datapoint in batch]).squeeze(1),
            pad_sequence([datapoint[4] for datapoint in batch], batch_first=True),
            pad_sequence([datapoint[5] for datapoint in batch], batch_first=True),
            pad_sequence([datapoint[6] for datapoint in batch], batch_first=True),
            torch.stack([datapoint[7] for datapoint in batch]).squeeze(),
            torch.stack([datapoint[8] for datapoint in batch]))
