import torch
import torch.multiprocessing
import wandb
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from TrainingInterfaces.Spectrogram_to_Embedding.StyleEmbedding import StyleEmbedding
from Utility.WarmupScheduler import WarmupScheduler
from Utility.path_to_transcript_dicts import *
from Utility.storage_config import MODELS_DIR
from Utility.utils import delete_old_checkpoints
from Utility.utils import get_most_recent_checkpoint
from Utility.utils import plot_progress_spec


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
               datasets,
               device,
               save_directory,
               batch_size,
               phase_1_steps,
               phase_2_steps,
               steps_per_checkpoint,
               lr,
               path_to_checkpoint,
               path_to_embed_model=os.path.join(MODELS_DIR, "Embedding", "embedding_function.pt"),
               resume=False,
               warmup_steps=4000,
               use_wandb=False):
    # ============
    # Preparations
    # ============
    steps = phase_1_steps + phase_2_steps
    net = net.to(device)

    style_embedding_function = StyleEmbedding().to(device)
    check_dict = torch.load(path_to_embed_model, map_location=device)
    style_embedding_function.load_state_dict(check_dict["style_emb_func"])
    style_embedding_function.eval()
    style_embedding_function.requires_grad_(False)

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

                    style_embedding = style_embedding_function(batch_of_spectrograms=batch[2].to(device),
                                                               batch_of_spectrogram_lengths=batch[3].to(device))

                    train_loss = train_loss + net(text_tensors=batch[0].to(device),
                                                  text_lengths=batch[1].to(device),
                                                  gold_speech=batch[2].to(device),
                                                  speech_lengths=batch[3].to(device),
                                                  gold_durations=batch[4].to(device),
                                                  gold_pitch=batch[6].to(device),  # mind the switched order
                                                  gold_energy=batch[5].to(device),  # mind the switched order
                                                  utterance_embedding=style_embedding,
                                                  lang_ids=batch[8].to(device),
                                                  return_mels=False)
                else:
                    # PHASE 2
                    style_embedding_function.eval()
                    style_embedding_of_gold, out_list_gold = style_embedding_function(
                        batch_of_spectrograms=batch[2].to(device),
                        batch_of_spectrogram_lengths=batch[3].to(device),
                        return_all_outs=True)

                    _train_loss, output_spectrograms = net(text_tensors=batch[0].to(device),
                                                           text_lengths=batch[1].to(device),
                                                           gold_speech=batch[2].to(device),
                                                           speech_lengths=batch[3].to(device),
                                                           gold_durations=batch[4].to(device),
                                                           gold_pitch=batch[6].to(device),  # mind the switched order
                                                           gold_energy=batch[5].to(device),  # mind the switched order
                                                           utterance_embedding=style_embedding_of_gold.detach(),
                                                           lang_ids=batch[8].to(device),
                                                           return_mels=True)
                    train_loss = train_loss + _train_loss
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

        if step % steps_per_checkpoint == 0 and step != 0:
            # ==============================
            # Enough steps for some insights
            # ==============================
            net.eval()
            style_embedding_function.eval()
            default_embedding = style_embedding_function(
                batch_of_spectrograms=datasets[0][0][2].unsqueeze(0).to(device),
                batch_of_spectrogram_lengths=datasets[0][0][3].unsqueeze(0).to(device)).squeeze()
            print(f"\nTotal Steps: {step}")
            print(f"Spectrogram Loss: {round(sum(train_losses_total) / len(train_losses_total), 3)}")
            if len(cycle_losses_total) != 0:
                print(f"Cycle Loss: {round(sum(cycle_losses_total) / len(cycle_losses_total), 3)}")
            torch.save({
                "model"       : net.state_dict(),
                "optimizer"   : optimizer.state_dict(),
                "scaler"      : grad_scaler.state_dict(),
                "scheduler"   : scheduler.state_dict(),
                "step_counter": step,
                "default_emb" : default_embedding,
            },
                os.path.join(save_directory, "checkpoint_{}.pt".format(step)))
            delete_old_checkpoints(save_directory, keep=5)
            path_to_most_recent_plot = plot_progress_spec(net=net,
                                                          device=device,
                                                          lang="en",
                                                          save_dir=save_directory,
                                                          step=step,
                                                          default_emb=default_embedding)
            if use_wandb:
                wandb.log({
                    "spectrogram_loss": round(sum(train_losses_total) / len(train_losses_total), 3),
                    "cycle_loss"      : round(sum(cycle_losses_total) / len(cycle_losses_total), 3) if len(
                        cycle_losses_total) != 0 else 0.0,
                    "Steps"           : step,
                    "progress_plot"   : wandb.Image(path_to_most_recent_plot)
                })
            train_losses_total = list()
            cycle_losses_total = list()
            net.train()
