import copy
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.multiprocessing import Process
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from Preprocessing.ArticulatoryCombinedTextFrontend import ArticulatoryCombinedTextFrontend
from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.Tacotron2 import Tacotron2
from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.TacotronDataset import TacotronDataset
from Utility.path_to_transcript_dicts import *
from Utility.utils import delete_old_checkpoints
from Utility.utils import get_most_recent_checkpoint


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume):
    torch.manual_seed(131714)
    random.seed(131714)
    torch.random.manual_seed(131714)

    assert resume_checkpoint is not None  # much better to start from a pretrained model that already knows diagonal alignment

    datasets = list()

    base_dir = os.path.join("Models", "Singe_Step_LAML")

    print("Preparing")
    cache_dir_english_nancy = os.path.join("Corpora", "meta_English_nancy")
    os.makedirs(cache_dir_english_nancy, exist_ok=True)
    datasets.append(TacotronDataset(build_path_to_transcript_dict_nancy(),
                                    cache_dir=cache_dir_english_nancy,
                                    lang="en",
                                    loading_processes=20,  # run this on a lonely server at night for the first time
                                    cut_silences=True,
                                    min_len_in_seconds=2,
                                    max_len_in_seconds=13))

    cache_dir_greek = os.path.join("Corpora", "meta_Greek")
    os.makedirs(cache_dir_greek, exist_ok=True)
    datasets.append(TacotronDataset(build_path_to_transcript_dict_css10el(),
                                    cache_dir=cache_dir_greek,
                                    lang="el",
                                    loading_processes=20,
                                    cut_silences=True,
                                    min_len_in_seconds=2,
                                    max_len_in_seconds=13))

    cache_dir_spanish = os.path.join("Corpora", "meta_Spanish")
    os.makedirs(cache_dir_spanish, exist_ok=True)
    datasets.append(TacotronDataset(build_path_to_transcript_dict_css10es(),
                                    cache_dir=cache_dir_spanish,
                                    lang="es",
                                    loading_processes=20,
                                    cut_silences=True,
                                    min_len_in_seconds=2,
                                    max_len_in_seconds=13))

    cache_dir_finnish = os.path.join("Corpora", "meta_Finnish")
    os.makedirs(cache_dir_finnish, exist_ok=True)
    datasets.append(TacotronDataset(build_path_to_transcript_dict_css10fi(),
                                    cache_dir=cache_dir_finnish,
                                    lang="fi",
                                    loading_processes=20,
                                    cut_silences=True,
                                    min_len_in_seconds=2,
                                    max_len_in_seconds=13))

    cache_dir_russian = os.path.join("Corpora", "meta_Russian")
    os.makedirs(cache_dir_russian, exist_ok=True)
    datasets.append(TacotronDataset(build_path_to_transcript_dict_css10ru(),
                                    cache_dir=cache_dir_russian,
                                    lang="ru",
                                    loading_processes=20,
                                    cut_silences=True,
                                    min_len_in_seconds=2,
                                    max_len_in_seconds=13))

    cache_dir_hungarian = os.path.join("Corpora", "meta_Hungarian")
    os.makedirs(cache_dir_hungarian, exist_ok=True)
    datasets.append(TacotronDataset(build_path_to_transcript_dict_css10hu(),
                                    cache_dir=cache_dir_hungarian,
                                    lang="hu",
                                    loading_processes=20,
                                    cut_silences=True,
                                    min_len_in_seconds=2,
                                    max_len_in_seconds=13))

    cache_dir_dutch = os.path.join("Corpora", "meta_Dutch")
    os.makedirs(cache_dir_dutch, exist_ok=True)
    datasets.append(TacotronDataset(build_path_to_transcript_dict_css10nl(),
                                    cache_dir=cache_dir_dutch,
                                    lang="nl",
                                    loading_processes=20,
                                    cut_silences=True,
                                    min_len_in_seconds=2,
                                    max_len_in_seconds=13))

    cache_dir_french = os.path.join("Corpora", "meta_French")
    os.makedirs(cache_dir_french, exist_ok=True)
    datasets.append(TacotronDataset(build_path_to_transcript_dict_css10fr(),
                                    cache_dir=cache_dir_french,
                                    lang="fr",
                                    loading_processes=20,
                                    cut_silences=True,
                                    min_len_in_seconds=2,
                                    max_len_in_seconds=13))

    if model_dir is not None:
        meta_save_dir = model_dir
    else:
        meta_save_dir = os.path.join(base_dir, "Tacotron2_MetaCheckpoint")
    os.makedirs(meta_save_dir, exist_ok=True)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"

    train_loop(net=Tacotron2(use_alignment_loss=False),
               device=torch.device("cuda"),
               datasets=datasets,
               batch_size=10,
               save_directory=meta_save_dir,
               steps=100000,
               steps_per_checkpoint=1000,
               lr=0.001,
               path_to_checkpoint=resume_checkpoint,
               resume=resume)


def train_loop(net,
               datasets,
               device,
               save_directory,
               batch_size,
               steps,
               steps_per_checkpoint,
               lr,
               path_to_checkpoint,
               resume=False):
    # ============
    # Preparations
    # ============
    net = net.to(device)
    train_loaders = list()
    train_iters = list()
    for dataset in datasets:
        train_loaders.append(DataLoader(batch_size=batch_size,
                                        dataset=dataset,
                                        drop_last=True,
                                        num_workers=2,
                                        pin_memory=False,
                                        shuffle=True,
                                        prefetch_factor=16,
                                        collate_fn=collate_and_pad,
                                        persistent_workers=True))
        train_iters.append(iter(train_loaders[-1]))
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1.0e-06, weight_decay=0.0)
    grad_scaler = GradScaler()
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
            if step_counter > steps:
                print("Desired steps already reached in loaded checkpoint.")
                return

    # =============================
    # Actual train loop starts here
    # =============================
    for step in tqdm(range(step_counter, steps)):
        net.train()
        optimizer.zero_grad()
        batches = []
        for index in range(len(train_iters)):
            # we get one batch for each task (i.e. language in this case)
            try:
                batches.append(next(train_iters[index]))
            except StopIteration:
                train_iters[index] = iter(datasets[index])
                batches.append(next(train_iters[index]))
        train_losses = list()
        for batch in batches:
            with autocast():
                # we sum the loss for each task, as we would do for the
                # second order regular MAML, but we do it only over one
                # step (i.e. iterations of inner loop = 1)
                train_losses.append(net(text=batch[0].to(device),
                                        text_lengths=batch[1].to(device),
                                        speech=batch[2].to(device),
                                        speech_lengths=batch[3].to(device),
                                        step=step,
                                        return_mels=False,
                                        return_loss_dict=False))
        # then we directly update our meta-parameters without
        # the need for any task specific parameters
        train_loss = sum(train_losses)
        train_losses_total.append(train_loss.item())
        optimizer.zero_grad()
        grad_scaler.scale(train_loss).backward()
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=False)
        grad_scaler.step(optimizer)
        grad_scaler.update()

        if step % steps_per_checkpoint == 0:
            # ==============================
            # Enough steps for some insights
            # ==============================
            net.eval()
            print(f"Total Loss: {round(sum(train_losses_total) / len(train_losses_total), 3)}")
            train_losses_total = list()
            torch.save({"model": net.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": grad_scaler.state_dict(),
                        "step_counter": step},
                       os.path.join(save_directory, "checkpoint_{}.pt".format(step)))
            delete_old_checkpoints(save_directory, keep=5)
            net_for_eval = copy.deepcopy(net)
            net_for_eval = net_for_eval.cpu()
            processes = list()
            for lang in ["en", "de", "el", "es", "fi", "ru", "hu", "nl", "fr"]:
                processes.append(Process(target=plot_attention,
                                         kwargs={"model": net_for_eval,
                                                 "lang": lang,
                                                 "att_dir": save_directory,
                                                 "step": step}))
                processes[-1].start()
            for process in processes:
                process.join()
            del net_for_eval


@torch.no_grad()
def plot_attention(model, lang, att_dir, step):
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
    text = tf.string_to_tensor(sentence)
    phones = tf.get_phone_string(sentence)
    _, _, att = model.inference(text_tensor=text)
    bin_att = mas(att.data.numpy())
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 9))
    ax[0].imshow(att.detach().numpy(), interpolation='nearest', aspect='auto', origin="lower")
    ax[1].imshow(bin_att, interpolation='nearest', aspect='auto', origin="lower")
    ax[1].set_xlabel("Inputs")
    ax[0].xaxis.set_visible(False)
    ax[0].set_ylabel("Outputs")
    ax[1].set_ylabel("Outputs")
    ax[1].set_xticks(range(len(att[0])))
    ax[1].set_xticklabels(labels=[phone for phone in phones])
    ax[0].set_title("Soft-Attention")
    ax[1].set_title("Hard-Attention")
    fig.tight_layout()
    if not os.path.exists(os.path.join(att_dir, "attention_plots")):
        os.makedirs(os.path.join(att_dir, "attention_plots"))
    fig.savefig(os.path.join(os.path.join(att_dir, "attention_plots"), f"{step}_{lang}.png"))
    fig.clf()
    plt.close()


def mas(attn_map):
    # assumes mel x text
    opt = np.zeros_like(attn_map)
    attn_map = np.log(attn_map)
    attn_map[0, 1:] = -np.inf
    log_p = np.zeros_like(attn_map)
    log_p[0, :] = attn_map[0, :]
    prev_ind = np.zeros_like(attn_map, dtype=np.int64)
    for i in range(1, attn_map.shape[0]):
        for j in range(attn_map.shape[1]):  # for each text dim
            prev_log = log_p[i - 1, j]
            prev_j = j
            if j - 1 >= 0 and log_p[i - 1, j - 1] >= log_p[i - 1, j]:
                prev_log = log_p[i - 1, j - 1]
                prev_j = j - 1
            log_p[i, j] = attn_map[i, j] + prev_log
            prev_ind[i, j] = prev_j
    # now backtrack
    curr_text_idx = attn_map.shape[1] - 1
    for i in range(attn_map.shape[0] - 1, -1, -1):
        opt[i, curr_text_idx] = 1
        curr_text_idx = prev_ind[i, curr_text_idx]
    opt[0, curr_text_idx] = 1
    return opt


def collate_and_pad(batch):
    # text, text_len, speech, speech_len
    return (pad_sequence([datapoint[0] for datapoint in batch], batch_first=True),
            torch.stack([datapoint[1] for datapoint in batch]).squeeze(1),
            pad_sequence([datapoint[2] for datapoint in batch], batch_first=True),
            torch.stack([datapoint[3] for datapoint in batch]).squeeze(1))
