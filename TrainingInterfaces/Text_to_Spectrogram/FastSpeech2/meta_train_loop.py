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
from TrainingInterfaces.Spectrogram_to_Embedding.StyleEmbedding import StyleEmbedding
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
               phase_1_steps,
               phase_2_steps,
               steps_per_checkpoint,
               lr,
               path_to_checkpoint,
               path_to_spk_embed_model="Models/Embedding/speaker_embedding_function.pt",
               path_to_emo_embed_model="Models/Embedding/emotion_embedding_function.pt",
               resume=False,
               warmup_steps=4000):
    # ============
    # Preparations
    # ============
    steps = phase_1_steps + phase_2_steps

    net = net.to(device)
    spk_style_embedding_function = StyleEmbedding().to(device)
    check_dict = torch.load(path_to_spk_embed_model, map_location=device)
    spk_style_embedding_function.load_state_dict(check_dict["style_emb_func"])
    spk_style_embedding_function.requires_grad_(False)

    emo_style_embedding_function = StyleEmbedding().to(device)
    check_dict = torch.load(path_to_emo_embed_model, map_location=device)
    emo_style_embedding_function.load_state_dict(check_dict["style_emb_func"])
    emo_style_embedding_function.requires_grad_(False)

    cycle_consistency_objective = torch.nn.MSELoss(reduction='mean')

    torch.multiprocessing.set_sharing_strategy('file_system')
    train_loaders = list()
    train_iters = list()
    for dataset in datasets:
        train_loaders.append(DataLoader(batch_size=batch_size,
                                        dataset=dataset,
                                        drop_last=True,
                                        num_workers=4,
                                        pin_memory=True,
                                        shuffle=True,
                                        prefetch_factor=5,
                                        collate_fn=collate_and_pad,
                                        persistent_workers=True))
        train_iters.append(iter(train_loaders[-1]))
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
    cycle_losses_total = list()
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
            # we get one batch for each task (i.e. language in this case) in a randomized order
            try:
                batch = next(train_iters[index])
                batches.append(batch)
            except StopIteration:
                train_iters[index] = iter(train_loaders[index])
                batch = next(train_iters[index])
                batches.append(batch)
        train_loss = 0.0
        cycle_loss = 0.0
        for batch in batches:
            with autocast():
                if step <= phase_1_steps:
                    # PHASE 1
                    # we sum the loss for each task, as we would do for the
                    # second order regular MAML, but we do it only over one
                    # step (i.e. iterations of inner loop = 1)
                    spk_style_embedding = spk_style_embedding_function(batch_of_spectrograms=batch[2].to(device),
                                                                       batch_of_spectrogram_lengths=batch[3].to(device))

                    emo_style_embedding = emo_style_embedding_function(batch_of_spectrograms=batch[2].to(device),
                                                                       batch_of_spectrogram_lengths=batch[3].to(device))

                    train_loss = train_loss + net(text_tensors=batch[0].to(device),
                                                  text_lengths=batch[1].to(device),
                                                  gold_speech=batch[2].to(device),
                                                  speech_lengths=batch[3].to(device),
                                                  gold_durations=batch[4].to(device),
                                                  gold_pitch=batch[6].to(device),  # mind the switched order
                                                  gold_energy=batch[5].to(device),  # mind the switched order
                                                  utterance_spk_embedding=spk_style_embedding,
                                                  utterance_emo_embedding=emo_style_embedding,
                                                  lang_ids=batch[8].to(device),
                                                  return_mels=False)
                else:
                    # PHASE 2
                    spk_style_embedding_of_gold = spk_style_embedding_function(batch_of_spectrograms=batch[2].to(device),
                                                                               batch_of_spectrogram_lengths=batch[3].to(device)).detach()

                    emo_style_embedding_of_gold = emo_style_embedding_function(batch_of_spectrograms=batch[2].to(device),
                                                                               batch_of_spectrogram_lengths=batch[3].to(device)).detach()

                    _train_loss, output_spectrograms = net(text_tensors=batch[0].to(device),
                                                           text_lengths=batch[1].to(device),
                                                           gold_speech=batch[2].to(device),
                                                           speech_lengths=batch[3].to(device),
                                                           gold_durations=batch[4].to(device),
                                                           gold_pitch=batch[6].to(device),  # mind the switched order
                                                           gold_energy=batch[5].to(device),  # mind the switched order
                                                           utterance_spk_embedding=spk_style_embedding,
                                                           utterance_emo_embedding=emo_style_embedding,
                                                           lang_ids=batch[8].to(device),
                                                           return_mels=True)
                    train_loss = train_loss + _train_loss

                    spk_style_embedding_of_predicted = spk_style_embedding_function(batch_of_spectrograms=output_spectrograms,
                                                                                    batch_of_spectrogram_lengths=batch[3].to(device))

                    emo_style_embedding_of_predicted = emo_style_embedding_function(batch_of_spectrograms=output_spectrograms,
                                                                                    batch_of_spectrogram_lengths=batch[3].to(device))

                    cycle_dist = cycle_consistency_objective(spk_style_embedding_of_predicted, spk_style_embedding_of_gold) * 300
                    cycle_dist += cycle_consistency_objective(emo_style_embedding_of_predicted, emo_style_embedding_of_gold) * 300

                    cycle_loss = cycle_loss + cycle_dist

        # then we directly update our meta-parameters without
        # the need for any task specific parameters
        train_losses_total.append(train_loss.item())
        if cycle_loss != 0.0:
            cycle_losses_total.append(cycle_loss.item())
        optimizer.zero_grad()
        train_loss = train_loss + cycle_loss
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
            spk_style_embedding_function.eval()
            emo_style_embedding_function.eval()
            default_spk_embedding = spk_style_embedding_function(batch_of_spectrograms=datasets[0][0][2].unsqueeze(0).to(device),
                                                                 batch_of_spectrogram_lengths=datasets[0][0][3].unsqueeze(0).to(device)).squeeze()
            default_emo_embedding = emo_style_embedding_function(batch_of_spectrograms=datasets[0][0][2].unsqueeze(0).to(device),
                                                                 batch_of_spectrogram_lengths=datasets[0][0][3].unsqueeze(0).to(device)).squeeze()
            print(f"\nTotal Steps: {step}")
            print(f"Total Loss: {round(sum(train_losses_total) / len(train_losses_total), 3)}")
            if len(cycle_losses_total) != 0:
                print(f"Cycle Loss: {round(sum(cycle_losses_total) / len(cycle_losses_total), 3)}")
            train_losses_total = list()
            cycle_losses_total = list()
            torch.save({
                "model"          : net.state_dict(),
                "optimizer"      : optimizer.state_dict(),
                "scaler"         : grad_scaler.state_dict(),
                "scheduler"      : scheduler.state_dict(),
                "step_counter"   : step,
                "default_spk_emb": default_spk_embedding,
                "default_emo_emb": default_emo_embedding,
                },
                os.path.join(save_directory, "checkpoint_{}.pt".format(step)))
            delete_old_checkpoints(save_directory, keep=5)
            plot_progress_spec(net=net,
                               device=device,
                               lang="en",
                               save_dir=save_directory,
                               step=step,
                               default_spk_emb=default_spk_embedding,
                               default_emo_emb=default_emo_embedding)
            net.train()
            spk_style_embedding_function.train()
            emo_style_embedding_function.train()


@torch.inference_mode()
def plot_progress_spec(net,
                       device,
                       save_dir,
                       step,
                       lang,
                       default_spk_emb,
                       default_emo_emb):
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
                                                   utterance_spk_embedding=default_spk_emb,
                                                   utterance_emo_embedding=default_emo_emb,
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
            ax.hlines(pitch_array[pitch_index] * 1000, xmin=xrange[0], xmax=xrange[1], color="blue", linestyles="solid", linewidth=0.5)
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
            None,
            torch.stack([datapoint[8] for datapoint in batch]))
