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

from Preprocessing.ArticulatoryCombinedTextFrontend import ArticulatoryCombinedTextFrontend
from Preprocessing.ArticulatoryCombinedTextFrontend import get_language_id
from TrainingInterfaces.Text_to_Spectrogram.AutoAligner.AlignerDataset import AlignerDataset
from TrainingInterfaces.Text_to_Spectrogram.AutoAligner.autoaligner_train_loop import train_loop as train_aligner
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeechDatasetLanguageID import FastSpeechDataset
from Utility.path_to_transcript_dicts import *
from Utility.utils import cumsum_durations
from Utility.utils import delete_old_checkpoints
from Utility.utils import get_most_recent_checkpoint


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume, find_faulty_samples_mode=False):
    torch.manual_seed(131714)
    random.seed(131714)
    torch.random.manual_seed(131714)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"

    datasets = list()

    base_dir = os.path.join("Models", "FastSpeech2_Meta")
    if model_dir is not None:
        meta_save_dir = model_dir
    else:
        meta_save_dir = base_dir
    os.makedirs(meta_save_dir, exist_ok=True)

    print("Preparing")

    datasets.append(prepare_corpus(transcript_dict=build_path_to_transcript_dict_mls_spanish(),
                                   corpus_dir=os.path.join("Corpora", "mls_spanish"),
                                   lang="es",
                                   ctc_selection=False))

    datasets.append(prepare_corpus(transcript_dict=build_path_to_transcript_dict_css10el(),
                                   corpus_dir=os.path.join("Corpora", "meta_Greek"),
                                   lang="el"))

    datasets.append(prepare_corpus(transcript_dict=build_path_to_transcript_dict_css10fi(),
                                   corpus_dir=os.path.join("Corpora", "meta_Finnish"),
                                   lang="fi"))

    datasets.append(prepare_corpus(transcript_dict=build_path_to_transcript_dict_css10ru(),
                                   corpus_dir=os.path.join("Corpora", "meta_Russian"),
                                   lang="ru"))

    datasets.append(prepare_corpus(transcript_dict=build_path_to_transcript_dict_css10hu(),
                                   corpus_dir=os.path.join("Corpora", "meta_Hungarian"),
                                   lang="hu"))

    datasets.append(prepare_corpus(transcript_dict=build_path_to_transcript_dict_css10fr(),
                                   corpus_dir=os.path.join("Corpora", "meta_French"),
                                   lang="fr"))

    datasets.append(prepare_corpus(transcript_dict=build_path_to_transcript_dict_att_hack(),
                                   corpus_dir=os.path.join("Corpora", "expressive_French"),
                                   lang="fr"))

    datasets.append(prepare_corpus(transcript_dict=build_path_to_transcript_dict_spanish_blizzard_train(),
                                   corpus_dir=os.path.join("Corpora", "spanish_blizzard"),
                                   lang="es"))

    datasets.append(prepare_corpus(transcript_dict=build_path_to_transcript_dict_mls_portuguese(),
                                   corpus_dir=os.path.join("Corpora", "mls_porto"),
                                   lang="pt",
                                   ctc_selection=False))

    datasets.append(prepare_corpus(transcript_dict=build_path_to_transcript_dict_mls_polish(),
                                   corpus_dir=os.path.join("Corpora", "mls_polish"),
                                   lang="pl",
                                   ctc_selection=False))

    datasets.append(prepare_corpus(transcript_dict=build_path_to_transcript_dict_css10es(),
                                   corpus_dir=os.path.join("Corpora", "meta_Spanish"),
                                   lang="es"))

    datasets.append(prepare_corpus(transcript_dict=build_path_to_transcript_dict_mls_french(),
                                   corpus_dir=os.path.join("Corpora", "mls_french"),
                                   lang="fr",
                                   ctc_selection=False))

    datasets.append(prepare_corpus(transcript_dict=build_path_to_transcript_dict_mls_italian(),
                                   corpus_dir=os.path.join("Corpora", "mls_italian"),
                                   lang="it",
                                   ctc_selection=False))


    if find_faulty_samples_mode:
        find_faulty_samples(net=FastSpeech2(lang_embs=100),
                            datasets=datasets,
                            device=torch.device("cuda"),
                            path_to_checkpoint=resume_checkpoint)
    else:
        train_loop(net=FastSpeech2(lang_embs=100),
                   device=torch.device("cuda"),
                   datasets=datasets,
                   batch_size=5,
                   save_directory=meta_save_dir,
                   steps=100000,
                   steps_per_checkpoint=1000,
                   lr=0.001,
                   path_to_checkpoint=resume_checkpoint,
                   resume=resume)


def prepare_corpus(transcript_dict, corpus_dir, lang, ctc_selection=True):
    """
    create an aligner dataset,
    fine-tune an aligner,
    create a fastspeech dataset,
    return it.

    Skips parts that have been done before.
    """
    aligner_dir = os.path.join(corpus_dir, "aligner")
    if not os.path.exists(os.path.join(aligner_dir, "aligner.pt")):
        train_aligner(train_dataset=AlignerDataset(transcript_dict, cache_dir=corpus_dir, lang=lang),
                      device=torch.device("cuda"),
                      save_directory=aligner_dir,
                      steps=(len(transcript_dict.keys()) / 32) * 2,  # 3 epochs worth of finetuning
                      batch_size=32,
                      path_to_checkpoint="Models/Aligner/aligner.pt",
                      fine_tune=True,
                      debug_img_path=aligner_dir,
                      resume=False)
    return FastSpeechDataset(transcript_dict,
                             acoustic_checkpoint_path=os.path.join(aligner_dir, "aligner.pt"),
                             cache_dir=corpus_dir,
                             device=torch.device("cuda"),
                             lang=lang,
                             ctc_selection=ctc_selection)


@torch.no_grad()
def find_faulty_samples(net,
                        datasets,
                        device,
                        path_to_checkpoint):
    net = net.to(device)
    torch.multiprocessing.set_sharing_strategy('file_system')
    check_dict = torch.load(os.path.join(path_to_checkpoint), map_location=device)
    net.load_state_dict(check_dict["model"])
    losses = list()
    index_pairs = list()
    for dataset_index in range(len(datasets)):
        for datapoint_index in tqdm(range(len(datasets[dataset_index]))):
            loss = net(text_tensors=datasets[dataset_index][datapoint_index][0].unsqueeze(0).to(device),
                       text_lengths=datasets[dataset_index][datapoint_index][1].to(device),
                       gold_speech=datasets[dataset_index][datapoint_index][2].unsqueeze(0).to(device),
                       speech_lengths=datasets[dataset_index][datapoint_index][3].to(device),
                       gold_durations=datasets[dataset_index][datapoint_index][4].unsqueeze(0).to(device),
                       gold_pitch=datasets[dataset_index][datapoint_index][6].unsqueeze(0).to(device),  # mind the switched order
                       gold_energy=datasets[dataset_index][datapoint_index][5].unsqueeze(0).to(device),  # mind the switched order
                       utterance_embedding=datasets[dataset_index][datapoint_index][7].unsqueeze(0).to(device),
                       lang_ids=datasets[dataset_index][datapoint_index][8].unsqueeze(0).to(device),
                       return_mels=False).squeeze()
            if torch.isnan(loss):
                print(f"CAREFUL, NAN DETECTED: {dataset_index}, {datapoint_index}")
            losses.append(loss.item())
            index_pairs.append((dataset_index, datapoint_index))
    loss_high_to_low = sorted(losses, reverse=True)
    print(loss_high_to_low)
    threshold = loss_high_to_low[1000]
    for index, loss in enumerate(losses):
        if loss > threshold:
            print(index_pairs[index])
            print(loss)


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
                                        pin_memory=True,
                                        shuffle=True,
                                        prefetch_factor=16,
                                        collate_fn=collate_and_pad,
                                        persistent_workers=True))
        train_iters.append(iter(train_loaders[-1]))
    default_embeddings = {"es": None, "el": None, "fi": None, "ru": None, "hu": None, "fr": None}
    for index, lang in enumerate(["es", "el", "fi", "ru", "hu", "fr"]):
        default_embedding = None
        for datapoint in datasets[index]:
            if default_embedding is None:
                default_embedding = datapoint[7].squeeze()
            else:
                default_embedding = default_embedding + datapoint[7].squeeze()
        default_embeddings[lang] = (default_embedding / len(datasets[index])).to(device)
    optimizer = torch.optim.RAdam(net.parameters(), lr=lr, eps=1.0e-06, weight_decay=0.0)
    # this is only included in more recent versions of torch, may need to upgrade torch if there's an error here
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
                                        lang_ids=batch[8].to(device),
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
                "model"       : net.state_dict(),
                "optimizer"   : optimizer.state_dict(),
                "scaler"      : grad_scaler.state_dict(),
                "step_counter": step,
                "default_emb" : default_embeddings["en"]
                },
                os.path.join(save_directory, "checkpoint_{}.pt".format(step)))
            delete_old_checkpoints(save_directory, keep=5)
            for lang in ["el", "es", "fi", "ru", "hu", "fr"]:
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
    ax.set_xticklabels(tf.get_phone_string(sentence))
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
