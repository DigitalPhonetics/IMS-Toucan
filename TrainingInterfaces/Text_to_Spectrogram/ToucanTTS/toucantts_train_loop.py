import os
import random
import time

import torch
import torch.multiprocessing
import wandb
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from TrainingInterfaces.Spectrogram_to_Embedding.StyleEmbedding import StyleEmbedding
from TrainingInterfaces.Text_to_Embedding.SentenceEmbeddingAdaptor import SentenceEmbeddingAdaptor
from TrainingInterfaces.Text_to_Spectrogram.ToucanTTS.SpectrogramDiscriminator import SpectrogramDiscriminator
from Utility.WarmupScheduler import ToucanWarmupScheduler as WarmupScheduler
from Utility.utils import delete_old_checkpoints
from Utility.utils import get_most_recent_checkpoint
from Utility.utils import plot_progress_spec_toucantts
from run_weight_averaging import average_checkpoints
from run_weight_averaging import get_n_recent_checkpoints_paths
from run_weight_averaging import load_net_toucan
from run_weight_averaging import save_model_for_use


def collate_and_pad(batch):
    # text, text_len, speech, speech_len, durations, energy, pitch, utterance condition, language_id, sentence string, filepath
    return (pad_sequence([datapoint[0] for datapoint in batch], batch_first=True),
            torch.stack([datapoint[1] for datapoint in batch]).squeeze(1),
            pad_sequence([datapoint[2] for datapoint in batch], batch_first=True),
            torch.stack([datapoint[3] for datapoint in batch]).squeeze(1),
            pad_sequence([datapoint[4] for datapoint in batch], batch_first=True),
            pad_sequence([datapoint[5] for datapoint in batch], batch_first=True),
            pad_sequence([datapoint[6] for datapoint in batch], batch_first=True),
            None,
            torch.stack([datapoint[8] for datapoint in batch]),
            [datapoint[9] for datapoint in batch],
            [datapoint[10] for datapoint in batch])


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
               postnet_start_steps,
               use_discriminator,
               sent_embs=None,
               random_emb=False,
               emovdb=False,
               replace_utt_sent_emb=False,
               word_embedding_extractor=None,
               use_adapted_embs=False,
               path_to_xvect=None,
               static_speaker_embed=False
               ):
    """
    see train loop arbiter for explanations of the arguments
    """
    net = net.to(device)
    if use_discriminator:
        discriminator = SpectrogramDiscriminator().to(device)

    if path_to_xvect is None:
        style_embedding_function = StyleEmbedding().to(device)
        check_dict = torch.load(path_to_embed_model, map_location=device)
        style_embedding_function.load_state_dict(check_dict["style_emb_func"])
        style_embedding_function.eval()
        style_embedding_function.requires_grad_(False)

    if use_adapted_embs:
        sentence_embedding_adaptor = SentenceEmbeddingAdaptor(sent_embed_dim=768, utt_embed_dim=64, speaker_embed_dim=512 if path_to_xvect is not None else None).to(device)
        check_dict = torch.load("Models/SentEmbAdaptor_01_EmoMulti_emoBERTcls_xvect/adaptor.pt", map_location=device)
        sentence_embedding_adaptor.load_state_dict(check_dict["model"])
        sentence_embedding_adaptor.eval()
        sentence_embedding_adaptor.requires_grad_(False)
    else:
        sentence_embedding_adaptor = None

    torch.multiprocessing.set_sharing_strategy('file_system')
    train_loader = DataLoader(batch_size=batch_size,
                              dataset=train_dataset,
                              drop_last=True,
                              num_workers=12 if os.cpu_count() > 12 else max(os.cpu_count() - 2, 1),
                              pin_memory=True,
                              shuffle=True,
                              prefetch_factor=2,
                              collate_fn=collate_and_pad,
                              persistent_workers=True)
    step_counter = 0
    if use_discriminator:
        optimizer = torch.optim.Adam(list(net.parameters()) + list(discriminator.parameters()), lr=lr)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = WarmupScheduler(optimizer, peak_lr=lr, warmup_steps=warmup_steps, max_steps=steps)
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
    start_time = time.time()
    while True:
        net.train()
        epoch += 1
        l1_losses_total = list()
        glow_losses_total = list()
        duration_losses_total = list()
        pitch_losses_total = list()
        energy_losses_total = list()
        generator_losses_total = list()
        discriminator_losses_total = list()
        sent_style_losses_total = list()

        for batch in tqdm(train_loader):
            train_loss = 0.0
            if path_to_xvect is not None:
                filepaths = batch[10]
                embeddings = []
                for path in filepaths:
                    embeddings.append(path_to_xvect[path])
                style_embedding = torch.stack(embeddings).to(device)
            else:
                style_embedding = style_embedding_function(batch_of_spectrograms=batch[2].to(device),
                                                        batch_of_spectrogram_lengths=batch[3].to(device))
            if sent_embs is not None:
                if emovdb:
                    filepaths = batch[10]
                    if random_emb:
                        emotions = [get_emotion_from_path(path) for path in filepaths]
                        sentence_embedding = torch.stack([random.choice(sent_embs[emotion]) for emotion in emotions]).to(device)
                    else:
                        sentence_embedding = torch.stack([sent_embs[path] for path in filepaths]).to(device)
                else:
                    sentences = batch[9]
                    sentence_embedding = torch.stack([sent_embs[sent] for sent in sentences]).to(device)
                if sentence_embedding_adaptor is not None:
                    sentence_embedding = sentence_embedding_adaptor(sentence_embedding=sentence_embedding,
                                                                    speaker_embedding=style_embedding if sentence_embedding_adaptor.speaker_embed_dim is not None else None,
                                                                    return_emb=True)
            else:
                sentence_embedding = None
            
            if static_speaker_embed:
                filepaths = batch[10]
                speaker_ids = torch.LongTensor([get_speakerid_from_path(path) for path in filepaths]).to(device)
            else:
                speaker_ids = None

            if replace_utt_sent_emb:
                style_embedding = sentence_embedding
            
            if word_embedding_extractor is not None:
                word_embedding, sentence_lens = word_embedding_extractor.encode(sentences=batch[9])
                word_embedding = word_embedding.to(device)
            else:
                word_embedding = None
                sentence_lens = None

            l1_loss, duration_loss, pitch_loss, energy_loss, glow_loss, sent_style_loss, generated_spectrograms = net(
                text_tensors=batch[0].to(device),
                text_lengths=batch[1].to(device),
                gold_speech=batch[2].to(device),
                speech_lengths=batch[3].to(device),
                gold_durations=batch[4].to(device),
                gold_pitch=batch[6].to(device),  # mind the switched order
                gold_energy=batch[5].to(device),  # mind the switched order
                utterance_embedding=style_embedding,
                speaker_id=speaker_ids,
                sentence_embedding=sentence_embedding,
                word_embedding=word_embedding,
                lang_ids=batch[8].to(device),
                return_mels=True,
                run_glow=step_counter > postnet_start_steps or fine_tune)

            if use_discriminator:
                discriminator_loss, generator_loss = calc_gan_outputs(real_spectrograms=batch[2].to(device),
                                                                      fake_spectrograms=generated_spectrograms,
                                                                      spectrogram_lengths=batch[3].to(device),
                                                                      discriminator=discriminator)
                if not torch.isnan(discriminator_loss):
                    train_loss = train_loss + discriminator_loss
                if not torch.isnan(generator_loss):
                    train_loss = train_loss + generator_loss
                discriminator_losses_total.append(discriminator_loss.item())
                generator_losses_total.append(generator_loss.item())

            if not torch.isnan(l1_loss):
                train_loss = train_loss + l1_loss
            if not torch.isnan(duration_loss):
                train_loss = train_loss + duration_loss
            if not torch.isnan(pitch_loss):
                train_loss = train_loss + pitch_loss
            if not torch.isnan(energy_loss):
                train_loss = train_loss + energy_loss
            if glow_loss is not None:
                if step_counter > postnet_start_steps and not torch.isnan(glow_loss):
                    train_loss = train_loss + glow_loss
            if sent_style_loss is not None:
                if not torch.isnan(sent_style_loss):
                    train_loss = train_loss + sent_style_loss
                sent_style_losses_total.append(sent_style_loss.item())

            l1_losses_total.append(l1_loss.item())
            duration_losses_total.append(duration_loss.item())
            pitch_losses_total.append(pitch_loss.item())
            energy_losses_total.append(energy_loss.item())
            if step_counter > postnet_start_steps + 500 or fine_tune:
                # start logging late so the magnitude difference is smaller
                glow_losses_total.append(glow_loss.item())

            optimizer.zero_grad()
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=False)
            optimizer.step()
            scheduler.step()
            step_counter += 1

        # EPOCH IS OVER
        net.eval()
        if path_to_xvect is None:
            style_embedding_function.eval()
        if sentence_embedding_adaptor is not None:
            sentence_embedding_adaptor.eval()
        if path_to_xvect is not None:
            default_embedding = path_to_xvect[train_dataset[0][10]]
        else:
            default_embedding = style_embedding_function(
                batch_of_spectrograms=train_dataset[0][2].unsqueeze(0).to(device),
                batch_of_spectrogram_lengths=train_dataset[0][3].unsqueeze(0).to(device)).squeeze()
        if replace_utt_sent_emb:
            if emovdb:
                if random_emb:
                    default_embedding = sent_embs["neutral"][0]
                else:
                    default_embedding = sent_embs[train_dataset[0][10]]
            else:
                default_embedding = sent_embs[train_dataset[0][9]]
        torch.save({
            "model"       : net.state_dict(),
            "optimizer"   : optimizer.state_dict(),
            "step_counter": step_counter,
            "scheduler"   : scheduler.state_dict(),
            "default_emb" : default_embedding,
        }, os.path.join(save_directory, "checkpoint_{}.pt".format(step_counter)))
        delete_old_checkpoints(save_directory, keep=5)

        print("\nEpoch:                  {}".format(epoch))
        print("Time elapsed:           {} Minutes".format(round((time.time() - start_time) / 60)))
        print("Reconstruction Loss:    {}".format(round(sum(l1_losses_total) / len(l1_losses_total), 3)))
        print("Steps:                  {}\n".format(step_counter))
        if use_wandb:
            wandb.log({
                "l1_loss"      : round(sum(l1_losses_total) / len(l1_losses_total), 5),
                "duration_loss": round(sum(duration_losses_total) / len(duration_losses_total), 5),
                "pitch_loss"   : round(sum(pitch_losses_total) / len(pitch_losses_total), 5),
                "energy_loss"  : round(sum(energy_losses_total) / len(energy_losses_total), 5),
                "glow_loss"    : round(sum(glow_losses_total) / len(glow_losses_total), 5) if len(glow_losses_total) != 0 else None,
                "sentence_style_loss": round(sum(sent_style_losses_total) / len(sent_style_losses_total), 5) if len(sent_style_losses_total) != 0 else None,
            }, step=step_counter)
            if use_discriminator:
                wandb.log({
                    "critic_loss"   : round(sum(discriminator_losses_total) / len(discriminator_losses_total), 5),
                    "generator_loss": round(sum(generator_losses_total) / len(generator_losses_total), 5),
                }, step=step_counter)

        try:
            path_to_most_recent_plot_before, \
            path_to_most_recent_plot_after = plot_progress_spec_toucantts(net,
                                                                          device,
                                                                          save_dir=save_directory,
                                                                          step=step_counter,
                                                                          lang=lang,
                                                                          default_emb=default_embedding,
                                                                          static_speaker_embed=static_speaker_embed,
                                                                          sent_embs=sent_embs,
                                                                          random_emb=random_emb,
                                                                          emovdb=emovdb,
                                                                          sent_emb_adaptor=sentence_embedding_adaptor,
                                                                          word_embedding_extractor=word_embedding_extractor,
                                                                          run_postflow=step_counter - 5 > postnet_start_steps)
            if use_wandb:
                wandb.log({
                    "progress_plot_before": wandb.Image(path_to_most_recent_plot_before)
                }, step=step_counter)
                if step_counter > postnet_start_steps or fine_tune:
                    wandb.log({
                        "progress_plot_after": wandb.Image(path_to_most_recent_plot_after)
                    }, step=step_counter)
        except IndexError:
            print("generating progress plots failed.")

        if step_counter > 3 * postnet_start_steps:
            # Run manual SWA (torch builtin doesn't work unfortunately due to the use of weight norm in the postflow)
            checkpoint_paths = get_n_recent_checkpoints_paths(checkpoint_dir=save_directory, n=2)
            averaged_model, default_embed = average_checkpoints(checkpoint_paths, load_func=load_net_toucan)
            save_model_for_use(model=averaged_model, default_embed=default_embed, name=os.path.join(save_directory, "best.pt"))
            check_dict = torch.load(os.path.join(save_directory, "best.pt"), map_location=device)
            net.load_state_dict(check_dict["model"])

        if step_counter > steps:
            return  # DONE

        net.train()


def calc_gan_outputs(real_spectrograms, fake_spectrograms, spectrogram_lengths, discriminator):
    # we have signals with lots of padding and different shapes, so we need to extract fixed size windows first.
    fake_window, real_window = get_random_window(fake_spectrograms, real_spectrograms, spectrogram_lengths)
    # now we have windows that are [batch_size, 200, 80]
    critic_loss = discriminator.calc_discriminator_loss(fake_window.unsqueeze(1), real_window.unsqueeze(1))
    generator_loss = discriminator.calc_generator_feedback(fake_window.unsqueeze(1), real_window.unsqueeze(1))
    critic_loss = critic_loss
    generator_loss = generator_loss
    return critic_loss, generator_loss


def get_random_window(generated_sequences, real_sequences, lengths):
    """
    This will return a randomized but consistent window of each that can be passed to the discriminator
    Suboptimal runtime because of a loop, should not be too bad, but a fix would be nice.
    """
    generated_windows = list()
    real_windows = list()
    window_size = 100  # corresponds to 1.6 seconds of audio in real time

    for end_index, generated, real in zip(lengths.squeeze(), generated_sequences, real_sequences):

        length = end_index
        real_spec_unpadded = real[:end_index]
        fake_spec_unpadded = generated[:end_index]
        while length < window_size:
            real_spec_unpadded = real_spec_unpadded.repeat((2, 1))
            fake_spec_unpadded = fake_spec_unpadded.repeat((2, 1))
            length = length * 2

        max_start = length - window_size
        start = random.randint(0, max_start)

        generated_windows.append(fake_spec_unpadded[start:start + window_size].unsqueeze(0))
        real_windows.append(real_spec_unpadded[start:start + window_size].unsqueeze(0))
    return torch.cat(generated_windows, dim=0), torch.cat(real_windows, dim=0)

def get_emotion_from_path(path):
    if "EmoV_DB" in path or "EmoVDB_Sam" in path:
        emotion = os.path.splitext(os.path.basename(path))[0].split("-16bit")[0].split("_")[0].lower()
        if emotion == "amused":
            emotion = "joy"
        if emotion == "sleepiness":
            raise NameError("emotion sleepiness should not be included")
    if "CREMA_D" in path:
        emotion = os.path.splitext(os.path.basename(path))[0].split('_')[2]
        if emotion == "ANG":
            emotion = "anger"
        if emotion == "DIS":
            emotion = "disgust"
        if emotion == "FEA":
            emotion = "fear"
        if emotion == "HAP":
            emotion = "joy"
        if emotion == "NEU":
            emotion = "neutral"
        if emotion == "SAD":
            emotion = "sadness"
    if "Emotional_Speech_Dataset_Singapore" in path:
        emotion = os.path.basename(os.path.dirname(path)).lower()
        if emotion == "angry":
            emotion = "anger"
        if emotion == "happy":
            emotion = "joy"
        if emotion == "sad":
            emotion = "sadness"
    if "RAVDESS" in path:
        emotion = os.path.splitext(os.path.basename(path))[0].split('-')[2]
        if emotion == "01":
            emotion = "neutral"
        if emotion == "02":
            raise NameError("emotion calm should not be included")
        if emotion == "03":
            emotion = "joy"
        if emotion == "04":
            emotion = "sadness"
        if emotion == "05":
            emotion = "anger"
        if emotion == "06":
            emotion = "fear"
        if emotion == "07":
            emotion = "disgust"
        if emotion == "08":
            emotion = "surprise"
    return emotion

def get_speakerid_from_path(path):
    if "Emotional_Speech_Dataset_Singapore" in path:
        speaker = os.path.split(os.path.split(os.path.dirname(path))[0])[1]
        if speaker == "0011":
            speaker_id = 0
        if speaker == "0012":
            speaker_id = 1
        if speaker == "0013":
            speaker_id = 2
        if speaker == "0014":
            speaker_id = 3
        if speaker == "0015":
            speaker_id = 4
        if speaker == "0016":
            speaker_id = 5
        if speaker == "0017":
            speaker_id = 6
        if speaker == "0018":
            speaker_id = 7
        if speaker == "0019":
            speaker_id = 8
        if speaker == "0020":
            speaker_id = 9
    return speaker_id
