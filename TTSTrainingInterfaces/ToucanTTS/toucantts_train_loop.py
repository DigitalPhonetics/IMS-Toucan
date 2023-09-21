import os
import time

import torch
import torch.multiprocessing
import wandb
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from EmbeddingModel.StyleEmbedding import StyleEmbedding
from Utility.WarmupScheduler import ToucanWarmupScheduler as WarmupScheduler
from Utility.utils import delete_old_checkpoints
from Utility.utils import get_most_recent_checkpoint
from Utility.utils import plot_progress_spec_toucantts
from run_weight_averaging import average_checkpoints
from run_weight_averaging import get_n_recent_checkpoints_paths
from run_weight_averaging import load_net_toucan
from run_weight_averaging import save_model_for_use


def collate_and_pad(batch):
    # text, text_len, speech, speech_len, durations, energy, pitch, utterance condition, language_id, speaker embedding
    # Assuming you have a list of tensors with shape [9, l, 1024]
    return (pad_sequence([datapoint[0] for datapoint in batch], batch_first=True),
            torch.stack([datapoint[1] for datapoint in batch]).squeeze(1),
            pad_sequence([datapoint[2] for datapoint in batch], batch_first=True),
            torch.stack([datapoint[3] for datapoint in batch]).squeeze(1),
            pad_sequence([datapoint[4] for datapoint in batch], batch_first=True),
            pad_sequence([datapoint[5] for datapoint in batch], batch_first=True),
            pad_sequence([datapoint[6] for datapoint in batch], batch_first=True),
            pad_sequence([datapoint[7] for datapoint in batch], batch_first=True),
            torch.stack([datapoint[8] for datapoint in batch]),
            torch.stack([datapoint[9] for datapoint in batch]))


def train_loop(net,
               train_dataset,
               device,
               save_directory,
               batch_size,
               lang,
               lr,
               warmup_steps,
               path_to_checkpoint,
               path_to_embed_model,
               fine_tune,
               resume,
               steps,
               use_wandb,
               train_embed
               ):
    """
    see train loop arbiter for explanations of the arguments
    """
    net = net.to(device)

    style_embedding_function = StyleEmbedding().to(device)
    if path_to_embed_model is not None:
        check_dict = torch.load(path_to_embed_model, map_location=device)
        style_embedding_function.load_state_dict(check_dict["style_emb_func"])
        if not train_embed:
            style_embedding_function.eval()
            style_embedding_function.requires_grad_(False)

    torch.multiprocessing.set_sharing_strategy('file_system')
    train_loader = DataLoader(batch_size=batch_size,
                              dataset=train_dataset,
                              drop_last=True,
                              num_workers=4 if os.cpu_count() > 4 else max(os.cpu_count() - 2, 1),
                              pin_memory=True,
                              shuffle=True,
                              prefetch_factor=2,
                              collate_fn=collate_and_pad,
                              persistent_workers=True)
    step_counter = 0

    optimizer = torch.optim.AdamW([p for name, p in net.named_parameters() if 'post_flow' not in name], lr=lr)
    flow_optimizer = torch.optim.AdamW(net.post_flow.parameters(), lr=lr)

    scheduler = WarmupScheduler(optimizer, peak_lr=lr, warmup_steps=warmup_steps, max_steps=steps)
    flow_scheduler = WarmupScheduler(flow_optimizer, peak_lr=lr, warmup_steps=warmup_steps // 4, max_steps=steps)

    epoch = 0
    if resume:
        path_to_checkpoint = get_most_recent_checkpoint(checkpoint_dir=save_directory)
    if path_to_checkpoint is not None:
        check_dict = torch.load(path_to_checkpoint, map_location=device)
        net.load_state_dict(check_dict["model"])
        if not fine_tune:
            optimizer.load_state_dict(check_dict["optimizer"])
            scheduler.load_state_dict(check_dict["scheduler"])
            flow_optimizer.load_state_dict(check_dict["flow_optimizer"])
            flow_scheduler.load_state_dict(check_dict["flow_scheduler"])
            step_counter = check_dict["step_counter"]
    start_time = time.time()
    while True:
        net.train()
        epoch += 1
        regression_losses_total = list()
        glow_losses_total = list()
        duration_losses_total = list()
        pitch_losses_total = list()
        energy_losses_total = list()

        for batch in tqdm(train_loader):
            run_glow = step_counter > warmup_steps * 3 or fine_tune
            train_loss = 0.0
            style_embedding = style_embedding_function(batch_of_feature_sequences=batch[7].to(device),
                                                       batch_of_feature_sequence_lengths=batch[3].to(device))
            utterance_embedding = torch.cat([style_embedding, batch[9].to(device)], dim=-1)
            regression_loss, glow_loss, duration_loss, pitch_loss, energy_loss, generated_features = net(
                text_tensors=batch[0].to(device),
                text_lengths=batch[1].to(device),
                gold_speech=batch[2].to(device),
                speech_lengths=batch[3].to(device),
                gold_durations=batch[4].to(device),
                gold_pitch=batch[6].to(device),  # mind the switched order
                gold_energy=batch[5].to(device),  # mind the switched order
                utterance_embedding=utterance_embedding,
                lang_ids=batch[8].to(device),
                return_feats=True,
                run_glow=run_glow
            )

            if step_counter % (warmup_steps // 4) == 0 and (path_to_embed_model is None or train_embed) and step_counter < warmup_steps * 2 and style_embedding_function.use_gst:
                # the computationally very expensive style token regularization loss to spread out the vectors
                print("calculating the style token regularization loss. This will take a while.")
                emb_reg_loss = style_embedding_function.style_encoder.calculate_ada4_regularization_loss()
                train_loss = train_loss + emb_reg_loss
            if not torch.isnan(regression_loss):
                train_loss = train_loss + regression_loss
            if glow_loss is not None:
                glow_losses_total.append(glow_loss.item())
                if not torch.isnan(glow_loss):
                    train_loss = train_loss + glow_loss
            else:
                glow_losses_total.append(0)
            if not torch.isnan(duration_loss):
                train_loss = train_loss + duration_loss
            if not torch.isnan(pitch_loss):
                train_loss = train_loss + pitch_loss
            if not torch.isnan(energy_loss):
                train_loss = train_loss + energy_loss

            regression_losses_total.append(regression_loss.item())
            duration_losses_total.append(duration_loss.item())
            pitch_losses_total.append(pitch_loss.item())
            energy_losses_total.append(energy_loss.item())

            optimizer.zero_grad()
            flow_optimizer.zero_grad()
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=False)
            optimizer.step()
            scheduler.step()
            if run_glow:
                flow_optimizer.step()
                flow_scheduler.step()
            step_counter += 1

        # EPOCH IS OVER
        net.eval()
        style_embedding_function.eval()
        default_embedding = torch.cat([style_embedding_function(
            batch_of_feature_sequences=train_dataset[0][7].unsqueeze(0).to(device),
            batch_of_feature_sequence_lengths=train_dataset[0][3].unsqueeze(0).to(device)).squeeze(),
                                       train_dataset[0][9].to(device)], dim=-1)
        torch.save({
            "model"         : net.state_dict(),
            "optimizer"     : optimizer.state_dict(),
            "step_counter"  : step_counter,
            "scheduler"     : scheduler.state_dict(),
            "flow_optimizer": flow_optimizer.state_dict(),
            "flow_scheduler": flow_scheduler.state_dict(),
            "default_emb"   : default_embedding,
            "config"        : net.config
        }, os.path.join(save_directory, "checkpoint_{}.pt".format(step_counter)))
        if path_to_embed_model is None or train_embed:
            torch.save({
                "style_emb_func": style_embedding_function.state_dict()
            }, os.path.join(save_directory, "embedding_function.pt"))
        delete_old_checkpoints(save_directory, keep=5)

        print(f"\nEpoch:                  {epoch}")
        print(f"Time elapsed:           {round((time.time() - start_time) / 60)} Minutes")
        print(f"Reconstruction Loss:    {round(sum(regression_losses_total) / len(regression_losses_total), 4)}")
        print(f"Steps:                  {step_counter}\n")

        if use_wandb:
            wandb.log({
                "regression_loss": round(sum(regression_losses_total) / len(regression_losses_total), 5),
                "glow_loss"      : round(sum(glow_losses_total) / len(glow_losses_total), 5),
                "duration_loss"  : round(sum(duration_losses_total) / len(duration_losses_total), 5),
                "pitch_loss"     : round(sum(pitch_losses_total) / len(pitch_losses_total), 5),
                "energy_loss"    : round(sum(energy_losses_total) / len(energy_losses_total), 5),
            }, step=step_counter)

        path_to_most_recent_plot = plot_progress_spec_toucantts(net,
                                                                device,
                                                                save_dir=save_directory,
                                                                step=step_counter,
                                                                lang=lang,
                                                                default_emb=default_embedding,
                                                                run_glow=run_glow)
        if use_wandb:
            wandb.log({
                "progress_plot": wandb.Image(path_to_most_recent_plot)
            }, step=step_counter)

        checkpoint_paths = get_n_recent_checkpoints_paths(checkpoint_dir=save_directory, n=2)
        averaged_model, default_embed = average_checkpoints(checkpoint_paths, load_func=load_net_toucan)
        save_model_for_use(model=averaged_model, default_embed=default_embed, name=os.path.join(save_directory, "best.pt"))
        if step_counter > steps * 4 / 5:
            # Run manual SWA (torch builtin doesn't work unfortunately due to the use of weight norm in the postflow)
            check_dict = torch.load(os.path.join(save_directory, "best.pt"), map_location=device)
            net.load_state_dict(check_dict["model"])

        if step_counter > steps:
            return  # DONE

        net.train()
        style_embedding_function.train()
