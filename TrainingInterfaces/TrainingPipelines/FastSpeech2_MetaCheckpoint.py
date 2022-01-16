import random

import librosa.display as lbd
import matplotlib.pyplot as plt
import torch
import torch.multiprocessing
import torch.multiprocessing
import torch.multiprocessing
import torch.multiprocessing
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from Preprocessing.ArticulatoryCombinedTextFrontend import ArticulatoryCombinedTextFrontend
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeechDataset import FastSpeechDataset
from Utility.path_to_transcript_dicts import *
from Utility.utils import cumsum_durations
from Utility.utils import delete_old_checkpoints
from Utility.utils import get_most_recent_checkpoint


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume):
    torch.manual_seed(131714)
    random.seed(131714)
    torch.random.manual_seed(131714)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"

    datasets = list()

    base_dir = os.path.join("Models", "FastSpeech2_MetaCheckpoint")

    print("Preparing")
    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_nancy(),
                                      acoustic_checkpoint_path="Models/Aligner/aligner.pt",
                                      cache_dir=os.path.join("Corpora", "Nancy"),
                                      device=torch.device("cuda"),
                                      lang="en"))

    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_karlsson(),
                                      acoustic_checkpoint_path="Models/Aligner/aligner.pt",
                                      cache_dir=os.path.join("Corpora", "Karlsson"),
                                      device=torch.device("cuda"),
                                      lang="de"))

    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_css10el(),
                                      acoustic_checkpoint_path="Models/Aligner/aligner.pt",
                                      cache_dir=os.path.join("Corpora", "meta_Greek"),
                                      device=torch.device("cuda"),
                                      lang="el"))

    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_css10es(),
                                      acoustic_checkpoint_path="Models/Aligner/aligner.pt",
                                      cache_dir=os.path.join("Corpora", "meta_Spanish"),
                                      device=torch.device("cuda"),
                                      lang="es"))

    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_css10fi(),
                                      acoustic_checkpoint_path="Models/Aligner/aligner.pt",
                                      cache_dir=os.path.join("Corpora", "meta_Finnish"),
                                      device=torch.device("cuda"),
                                      lang="fi"))

    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_css10ru(),
                                      acoustic_checkpoint_path="Models/Aligner/aligner.pt",
                                      cache_dir=os.path.join("Corpora", "meta_Russian"),
                                      device=torch.device("cuda"),
                                      lang="ru"))

    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_css10hu(),
                                      acoustic_checkpoint_path="Models/Aligner/aligner.pt",
                                      cache_dir=os.path.join("Corpora", "meta_Hungarian"),
                                      device=torch.device("cuda"),
                                      lang="hu"))

    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_css10nl(),
                                      acoustic_checkpoint_path="Models/Aligner/aligner.pt",
                                      cache_dir=os.path.join("Corpora", "meta_Dutch"),
                                      device=torch.device("cuda"),
                                      lang="nl"))

    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_css10fr(),
                                      acoustic_checkpoint_path="Models/Aligner/aligner.pt",
                                      cache_dir=os.path.join("Corpora", "meta_French"),
                                      device=torch.device("cuda"),
                                      lang="fr"))

    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_ljspeech(),
                                      acoustic_checkpoint_path="Models/Aligner/aligner.pt",
                                      cache_dir=os.path.join("Corpora", "LJSpeech"),
                                      device=torch.device("cuda"),
                                      lang="en"))

    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_hokuspokus(),
                                      acoustic_checkpoint_path="Models/Aligner/aligner.pt",
                                      cache_dir=os.path.join("Corpora", "Hokus"),
                                      device=torch.device("cuda"),
                                      lang="de"))

    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_thorsten(),
                                      acoustic_checkpoint_path="Models/Aligner/aligner.pt",
                                      cache_dir=os.path.join("Corpora", "Thorsten"),
                                      device=torch.device("cuda"),
                                      lang="de"))

    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_libritts(),
                                      acoustic_checkpoint_path="Models/Aligner/aligner.pt",
                                      cache_dir=os.path.join("Corpora", "libri"),
                                      device=torch.device("cuda"),
                                      lang="en"))

    # datasets.append(FastSpeechDataset(build_path_to_transcript_dict_nvidia_hifitts(),
    #                                  acoustic_checkpoint_path="Models/Aligner/aligner.pt",
    #                                  cache_dir=os.path.join("Corpora", "hifiTTS"),
    #                                  device=torch.device("cuda"),
    #                                  lang="en"))

    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_att_hack(),
                                      acoustic_checkpoint_path="Models/Aligner/aligner.pt",
                                      cache_dir=os.path.join("Corpora", "expressive_French"),
                                      device=torch.device("cuda"),
                                      lang="fr"))

    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_vctk(),
                                      acoustic_checkpoint_path="Models/Aligner/aligner.pt",
                                      cache_dir=os.path.join("Corpora", "vctk"),
                                      device=torch.device("cuda"),
                                      lang="en"))

    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_fluxsing(),
                                      acoustic_checkpoint_path="Models/Aligner/aligner.pt",
                                      cache_dir=os.path.join("Corpora", "flux_sing"),
                                      device=torch.device("cuda"),
                                      lang="en",
                                      ctc_selection=False))

    if model_dir is not None:
        meta_save_dir = model_dir
    else:
        meta_save_dir = base_dir
    os.makedirs(meta_save_dir, exist_ok=True)

    train_loop(net=FastSpeech2(),
               device=torch.device("cuda"),
               datasets=datasets,
               batch_size=5,
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
    torch.multiprocessing.set_sharing_strategy('file_system')
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
    default_embeddings = {"en": None, "de": None, "el": None, "es": None, "fi": None, "ru": None, "hu": None, "nl": None, "fr": None}
    for index, lang in enumerate(["en", "de", "el", "es", "fi", "ru", "hu", "nl", "fr"]):
        default_embedding = None
        for datapoint in datasets[index]:
            if default_embedding is None:
                default_embedding = datapoint[7].squeeze()
            else:
                default_embedding = default_embedding + datapoint[7].squeeze()
        default_embeddings[lang] = (default_embedding / len(datasets[index])).to(device)
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
        batches = []
        for index in range(len(datasets)):
            # we get one batch for each task (i.e. language in this case)
            try:
                batch = next(train_iters[index])
                batches.append(batch)
            except StopIteration:
                train_iters[index] = iter(train_loaders[index])
                batch = next(train_iters[index])
                batches.append(batch)
        train_losses = list()
        for batch in batches:
            with autocast():
                # we sum the loss for each task, as we would do for the
                # second order regular MAML, but we do it only over one
                # step (i.e. iterations of inner loop = 1)
                train_losses.append(net(text_tensors=batch[0].to(device),
                                        text_lengths=batch[1].to(device),
                                        gold_speech=batch[2].to(device),
                                        speech_lengths=batch[3].to(device),
                                        gold_durations=batch[4].to(device),
                                        gold_pitch=batch[6].to(device),  # mind the switched order
                                        gold_energy=batch[5].to(device),  # mind the switched order
                                        utterance_embedding=batch[7].to(device),
                                        return_mels=False))
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
            torch.save({
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": grad_scaler.state_dict(),
                "step_counter": step,
                "default_emb": default_embeddings["en"]
            },
                os.path.join(save_directory, "checkpoint_{}.pt".format(step)))
            delete_old_checkpoints(save_directory, keep=5)
            for lang in ["en", "de", "el", "es", "fi", "ru", "hu", "nl", "fr"]:
                plot_progress_spec(net=net,
                                   device=device,
                                   lang=lang,
                                   save_dir=save_directory,
                                   step=step,
                                   utt_embeds=default_embeddings)


@torch.no_grad()
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
    phoneme_vector = tf.string_to_tensor(sentence).squeeze(0).to(device)
    spec, durations, *_ = net.inference(text=phoneme_vector, return_duration_pitch_energy=True, utterance_embedding=default_embed)
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
    plt.savefig(os.path.join(os.path.join(save_dir, "spec"), f"{step}_{lang}.png"))
    plt.clf()
    plt.close()


def collate_and_pad(batch):
    # text, text_len, speech, speech_len, durations, energy, pitch, utterance condition
    return (pad_sequence([datapoint[0] for datapoint in batch], batch_first=True),
            torch.stack([datapoint[1] for datapoint in batch]).squeeze(1),
            pad_sequence([datapoint[2] for datapoint in batch], batch_first=True),
            torch.stack([datapoint[3] for datapoint in batch]).squeeze(1),
            pad_sequence([datapoint[4] for datapoint in batch], batch_first=True),
            pad_sequence([datapoint[5] for datapoint in batch], batch_first=True),
            pad_sequence([datapoint[6] for datapoint in batch], batch_first=True),
            torch.stack([datapoint[7] for datapoint in batch]).squeeze())
