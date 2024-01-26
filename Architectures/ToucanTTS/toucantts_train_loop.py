import os
import time

import torch
import torch.multiprocessing
import wandb
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from Preprocessing.AudioPreprocessor import AudioPreprocessor
from Preprocessing.EnCodecAudioPreprocessor import CodecAudioPreprocessor
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
    return (pad_sequence([datapoint[0] for datapoint in batch], batch_first=True).float(),
            torch.stack([datapoint[1] for datapoint in batch]).squeeze(1),
            [datapoint[2] for datapoint in batch],
            torch.stack([datapoint[3] for datapoint in batch]).squeeze(1),
            pad_sequence([datapoint[4] for datapoint in batch], batch_first=True),
            pad_sequence([datapoint[5] for datapoint in batch], batch_first=True),
            pad_sequence([datapoint[6] for datapoint in batch], batch_first=True),
            None,
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
               fine_tune,
               resume,
               steps,
               use_wandb,
               train_sampler,
               gpu_count,
               steps_per_checkpoint
               ):
    """
    see train loop arbiter for explanations of the arguments
    """
    net = net.to(device)
    if gpu_count > 1:
        rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
    if steps_per_checkpoint is None:
        steps_per_checkpoint = len(train_dataset) // batch_size

    if steps < warmup_steps * 5:
        print(f"too much warmup given the amount of steps, reducing warmup to {warmup_steps} steps")
        warmup_steps = steps // 5

    torch.multiprocessing.set_sharing_strategy('file_system')
    batch_sampler_train = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_sampler=batch_sampler_train,
                              num_workers=0,
                              pin_memory=True,
                              prefetch_factor=None,
                              collate_fn=collate_and_pad)
    ap = CodecAudioPreprocessor(input_sr=-1, device=device)
    spec_extractor = AudioPreprocessor(input_sr=16000, output_sr=16000, device=device)

    step_counter = 0

    if isinstance(net, torch.nn.parallel.DistributedDataParallel):
        model = net.module
    else:
        model = net
    optimizer = torch.optim.Adam([p for name, p in model.named_parameters() if 'post_flow' not in name], lr=lr)
    flow_optimizer = torch.optim.Adam(model.post_flow.parameters(), lr=lr * 8)

    scheduler = WarmupScheduler(optimizer, peak_lr=lr, warmup_steps=warmup_steps, max_steps=steps)
    flow_scheduler = WarmupScheduler(flow_optimizer, peak_lr=lr, warmup_steps=(warmup_steps // 4), max_steps=steps)

    epoch = 0
    if resume:
        path_to_checkpoint = get_most_recent_checkpoint(checkpoint_dir=save_directory)
    if path_to_checkpoint is not None:
        check_dict = torch.load(path_to_checkpoint, map_location=device)
        model.load_state_dict(check_dict["model"])
        if not fine_tune:
            optimizer.load_state_dict(check_dict["optimizer"])
            scheduler.load_state_dict(check_dict["scheduler"])
            flow_optimizer.load_state_dict(check_dict["flow_optimizer"])
            flow_scheduler.load_state_dict(check_dict["flow_scheduler"])
            step_counter = check_dict["step_counter"]
    start_time = time.time()
    regression_losses_total = list()
    glow_losses_total = list()
    duration_losses_total = list()
    pitch_losses_total = list()
    energy_losses_total = list()
    while True:
        net.train()
        epoch += 1
        for batch in tqdm(train_loader):

            text_tensors = batch[0].to(device)
            text_lengths = batch[1].squeeze().to(device)
            speech_indexes = batch[2]
            speech_lengths = batch[3].squeeze().to(device)
            gold_durations = batch[4].to(device)
            gold_pitch = batch[6].to(device)  # mind the switched order
            gold_energy = batch[5].to(device)  # mind the switched order
            lang_ids = batch[8].squeeze(1).to(device)

            speech_batch = list()  # I wish this could be done in the collate function or in the getitem, but using DL models in multiprocessing on very large datasets causes just way too many issues.
            for speech_sample in speech_indexes:
                with torch.inference_mode():
                    wave = ap.indexes_to_audio(speech_sample.int().to(device)).detach()
                    mel = spec_extractor.audio_to_mel_spec_tensor(wave, explicit_sampling_rate=16000).transpose(0, 1).detach().cpu()
                gold_speech_sample = mel.clone()
                speech_batch.append(gold_speech_sample)
            gold_speech = pad_sequence(speech_batch, batch_first=True).to(device)

            run_glow = step_counter > (warmup_steps * 2) or fine_tune

            train_loss = 0.0
            utterance_embedding = batch[9].to(device)
            regression_loss, glow_loss, duration_loss, pitch_loss, energy_loss = net(
                text_tensors=text_tensors,
                text_lengths=text_lengths,
                gold_speech=gold_speech,
                speech_lengths=speech_lengths,
                gold_durations=gold_durations,
                gold_pitch=gold_pitch,
                gold_energy=gold_energy,
                utterance_embedding=utterance_embedding,
                lang_ids=lang_ids,
                return_feats=False,
                run_glow=run_glow
            )

            if torch.isnan(regression_loss) or torch.isnan(duration_loss) or torch.isnan(pitch_loss) or torch.isnan(energy_loss):
                print("One of the losses turned to NaN! Skipping this batch ...")
                continue

            train_loss = train_loss + duration_loss
            train_loss = train_loss + pitch_loss
            train_loss = train_loss + energy_loss
            train_loss = train_loss + regression_loss

            regression_losses_total.append(regression_loss.item())
            duration_losses_total.append(duration_loss.item())
            pitch_losses_total.append(pitch_loss.item())
            energy_losses_total.append(energy_loss.item())

            if glow_loss is not None:

                if torch.isnan(glow_loss):
                    print("Glow loss turned to NaN! Skipping this batch ...")
                    continue

                if glow_loss < 0.0:
                    glow_losses_total.append(glow_loss.item())
                else:
                    glow_losses_total.append(0.1)

                train_loss = train_loss + glow_loss
            else:
                glow_losses_total.append(0)

            optimizer.zero_grad()
            flow_optimizer.zero_grad()
            if type(train_loss) is float:
                print("There is no loss for this step! Skipping ...")
                continue
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_([p for name, p in model.named_parameters() if 'post_flow' not in name], 1.0, error_if_nonfinite=False)
            optimizer.step()
            scheduler.step()
            if glow_loss is not None:
                torch.nn.utils.clip_grad_norm_(model.post_flow.parameters(), 1.0, error_if_nonfinite=False)
                flow_optimizer.step()
                flow_scheduler.step()
            step_counter += 1
            if step_counter % steps_per_checkpoint == 0:
                # evaluation interval is happening
                if rank == 0:
                    net.eval()
                    default_embedding = train_dataset[0][9].to(device)
                    torch.save({
                        "model"         : model.state_dict(),
                        "optimizer"     : optimizer.state_dict(),
                        "step_counter"  : step_counter,
                        "scheduler"     : scheduler.state_dict(),
                        "flow_optimizer": flow_optimizer.state_dict(),
                        "flow_scheduler": flow_scheduler.state_dict(),
                        "default_emb"   : default_embedding,
                        "config"        : model.config
                    }, os.path.join(save_directory, "checkpoint_{}.pt".format(step_counter)))

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
                            "learning_rate"  : optimizer.param_groups[0]['lr']
                        }, step=step_counter)
                    regression_losses_total = list()
                    glow_losses_total = list()
                    duration_losses_total = list()
                    pitch_losses_total = list()
                    energy_losses_total = list()

                    path_to_most_recent_plot = plot_progress_spec_toucantts(model,
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

                    checkpoint_paths = get_n_recent_checkpoints_paths(checkpoint_dir=save_directory, n=1)
                    averaged_model, default_embed = average_checkpoints(checkpoint_paths, load_func=load_net_toucan)
                    save_model_for_use(model=averaged_model, default_embed=default_embed, name=os.path.join(save_directory, "best.pt"))

                    if step_counter > steps:
                        return  # DONE

                    net.train()

        print("\n\n\nEPOCH COMPLETE\n\n\n")
