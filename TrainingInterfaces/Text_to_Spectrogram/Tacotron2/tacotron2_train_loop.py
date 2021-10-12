import os
import time

import matplotlib.pyplot as plt
import torch
import torch.multiprocessing
import torch.nn.functional as F
from speechbrain.pretrained import EncoderClassifier
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from Preprocessing.ArticulatoryCombinedTextFrontend import ArticulatoryCombinedTextFrontend
from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.AlignmentLoss import binarize_attention_parallel
from Utility.utils import delete_old_checkpoints
from Utility.utils import get_most_recent_checkpoint


def plot_attention(model, lang, device, speaker_embedding, att_dir, step, language_id):
    tf = ArticulatoryCombinedTextFrontend(language=lang)
    sentence = ""
    if lang == "en":
        sentence = "This is a complex sentence, it even has a pause!"
    elif lang == "de":
        sentence = "Dies ist ein komplexer Satz, er hat sogar eine Pause!"
    text = tf.string_to_tensor(sentence).to(device)
    phones = tf.get_phone_string(sentence)
    model.eval()
    _, _, att, att_loc, att_for = model.inference(text_tensor=text, speaker_embeddings=speaker_embedding, language_id=language_id, return_atts=True)
    att, att_loc, att_for = att.to("cpu"), att_loc.to("cpu"), att_for.to("cpu")
    model.train()
    del tf
    bin_att = binarize_attention_parallel(att.unsqueeze(0).unsqueeze(1),
                                          in_lens=torch.LongTensor([len(text)]),
                                          out_lens=torch.LongTensor([len(att)])).squeeze(0).squeeze(0).detach().numpy()
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 9))
    ax[0][0].imshow(att.detach().numpy(), interpolation='nearest', aspect='auto', origin="lower")
    ax[1][0].imshow(bin_att, interpolation='nearest', aspect='auto', origin="lower")

    ax[0][1].imshow(att_loc.detach().numpy(), interpolation='nearest', aspect='auto', origin="lower")
    ax[1][1].imshow(att_for.detach().numpy(), interpolation='nearest', aspect='auto', origin="lower")

    ax[1][0].set_xlabel("Inputs")
    ax[1][1].set_xlabel("Inputs")
    ax[0][0].xaxis.set_visible(False)
    ax[0][1].xaxis.set_visible(False)

    ax[1][1].yaxis.set_visible(False)
    ax[0][1].yaxis.set_visible(False)

    ax[0][0].set_ylabel("Outputs")
    ax[1][0].set_ylabel("Outputs")
    ax[1][0].set_xticks(range(len(att[0])))
    ax[1][1].set_xticks(range(len(att[0])))
    del att
    del att_loc
    del att_for
    ax[1][0].set_xticklabels(labels=[phone for phone in phones])
    ax[1][1].set_xticklabels(labels=[phone for phone in phones])
    ax[0][0].set_title("Soft-Combined-Attention")
    ax[1][0].set_title("Hard-Combined-Attention")
    ax[0][1].set_title("Location-Attention")
    ax[1][1].set_title("Forward-Attention")
    fig.tight_layout()

    if not os.path.exists(os.path.join(att_dir, "attention_plots")):
        os.makedirs(os.path.join(att_dir, "attention_plots"))
    fig.savefig(os.path.join(os.path.join(att_dir, "attention_plots"), str(step) + ".png"))
    fig.clf()
    plt.close()


def collate_and_pad(batch):
    max_text = max([datapoint[1] for datapoint in batch])
    max_spec = max([datapoint[3] for datapoint in batch])
    if type(batch[0][-1]) is int:
        if len(batch[0]) == 7:
            # text, text_len, speech, speech_len, speaker_emb, prior, language_id
            return (pad_sequence([datapoint[0] for datapoint in batch], batch_first=True),
                    torch.stack([datapoint[1] for datapoint in batch]).squeeze(1),
                    pad_sequence([datapoint[2] for datapoint in batch], batch_first=True),
                    torch.stack([datapoint[3] for datapoint in batch]).squeeze(1),
                    torch.stack([datapoint[4] for datapoint in batch]),
                    torch.stack([F.pad(datapoint[5], [0, max_text - datapoint[1], 0, max_spec - datapoint[3]]) for datapoint in batch]),
                    torch.stack([torch.LongTensor([datapoint[6]]) for datapoint in batch]).squeeze(1))
        else:
            # text, text_len, speech, speech_len, prior, language_id
            return (pad_sequence([datapoint[0] for datapoint in batch], batch_first=True),
                    torch.stack([datapoint[1] for datapoint in batch]).squeeze(1),
                    pad_sequence([datapoint[2] for datapoint in batch], batch_first=True),
                    torch.stack([datapoint[3] for datapoint in batch]).squeeze(1),
                    torch.stack([F.pad(datapoint[4], [0, max_text - datapoint[1], 0, max_spec - datapoint[3]]) for datapoint in batch]),
                    torch.stack([torch.LongTensor([datapoint[5]]) for datapoint in batch]).squeeze(1))
    else:
        if len(batch[0]) == 6:
            # text, text_len, speech, speech_len, speaker_emb, prior
            return (pad_sequence([datapoint[0] for datapoint in batch], batch_first=True),
                    torch.stack([datapoint[1] for datapoint in batch]).squeeze(1),
                    pad_sequence([datapoint[2] for datapoint in batch], batch_first=True),
                    torch.stack([datapoint[3] for datapoint in batch]).squeeze(1),
                    torch.stack([datapoint[4] for datapoint in batch]),
                    torch.stack([F.pad(datapoint[5], [0, max_text - datapoint[1], 0, max_spec - datapoint[3]]) for datapoint in batch]),
                    )
        else:
            # text, text_len, speech, speech_len, prior
            return (pad_sequence([datapoint[0] for datapoint in batch], batch_first=True),
                    torch.stack([datapoint[1] for datapoint in batch]).squeeze(1),
                    pad_sequence([datapoint[2] for datapoint in batch], batch_first=True),
                    torch.stack([datapoint[3] for datapoint in batch]).squeeze(1),
                    torch.stack([F.pad(datapoint[4], [0, max_text - datapoint[1], 0, max_spec - datapoint[3]]) for datapoint in batch]),
                    )


def train_loop(net,
               train_dataset,
               device,
               save_directory,
               batch_size=22,
               steps=100000,
               epochs_per_save=2,
               use_speaker_embedding=False,
               lang="en",
               lr=0.001,
               path_to_checkpoint=None,
               fine_tune=False,
               multi_ling=False,
               freeze_encoder_until=None,
               freeze_decoder_until=None,
               freeze_embedding_until=None,
               collapse_margin=5.0,  # be wary of loss scheduling
               resume=False):
    """
    Args:
        resume: whether to resume from the most recent checkpoint
        collapse_margin: margin in which the loss may increase in one epoch without triggering the soft-reset
        steps: How many steps to train
        lr: The initial learning rate for the optimiser
        path_to_checkpoint: reloads a checkpoint to continue training from there
        fine_tune: whether to load everything from a checkpoint, or only the model parameters
        lang: language of the synthesis
        use_speaker_embedding: whether to expect speaker embeddings
        net: Model to train
        train_dataset: Pytorch Dataset Object for train data
        device: Device to put the loaded tensors on
        save_directory: Where to save the checkpoints
        batch_size: How many elements should be loaded at once
        epochs_per_save: how many epochs to train in between checkpoints
        freeze_embedding_until: which step to unfreeze embedding function weights
        freeze_decoder_until: which step to unfreeze decoder weights
        freeze_encoder_until: which step to unfreeze encoder weights. Careful, this subsumes embedding weights
        multi_ling: whether to use language IDs for language embeddings
    """
    net = net.to(device)
    if freeze_decoder_until is not None and freeze_decoder_until > 0:
        for param in net.dec.parameters():
            param.requires_grad = False
    if freeze_encoder_until is not None and freeze_encoder_until > 0:
        for param in net.enc.parameters():
            param.requires_grad = False
    if freeze_embedding_until is not None and freeze_embedding_until > 0:
        for param in net.enc.embed.parameters():
            param.requires_grad = False
    previous_error = 999999  # tacotron can collapse sometimes and requires soft-resets. This is to detect collapses.
    train_loader = DataLoader(batch_size=batch_size,
                              dataset=train_dataset,
                              drop_last=True,
                              num_workers=8,
                              pin_memory=False,
                              shuffle=True,
                              prefetch_factor=8,
                              collate_fn=collate_and_pad,
                              persistent_workers=True)
    if multi_ling:
        reference_language_id = 1
    else:
        reference_language_id = None
    if use_speaker_embedding:
        reference_speaker_embedding_for_att_plot = torch.Tensor(train_dataset[0][4]).to(device)
        speaker_embedding_func = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                                run_opts={"device": str(device)},
                                                                savedir="Models/SpeakerEmbedding/speechbrain_speaker_embedding_ecapa")
    else:
        reference_speaker_embedding_for_att_plot = None
    step_counter = 0
    epoch = 0
    net.train()
    if fine_tune:
        lr = lr * 0.01
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1.0e-06, weight_decay=0.0)
    scaler = GradScaler()
    if resume:
        path_to_checkpoint = get_most_recent_checkpoint(checkpoint_dir=save_directory)
    if path_to_checkpoint is not None:
        check_dict = torch.load(os.path.join(path_to_checkpoint), map_location=device)
        net.load_state_dict(check_dict["model"])
        if not fine_tune:
            optimizer.load_state_dict(check_dict["optimizer"])
            step_counter = check_dict["step_counter"]
            scaler.load_state_dict(check_dict["scaler"])
    start_time = time.time()
    while True:
        cumulative_loss_dict = dict()
        epoch += 1
        optimizer.zero_grad()
        train_losses_this_epoch = list()
        for batch in tqdm(train_loader):
            with autocast():
                if multi_ling:
                    if not use_speaker_embedding:
                        train_loss, loss_dict = net(text=batch[0].to(device),
                                                    text_lengths=batch[1].to(device),
                                                    speech=batch[2].to(device),
                                                    speech_lengths=batch[3].to(device),
                                                    step=step_counter,
                                                    speaker_embeddings=None,
                                                    prior=batch[4].to(device),
                                                    language_id=batch[5].to(device),
                                                    return_loss_dict=True)
                    else:
                        train_loss, predicted_mels, loss_dict = net(text=batch[0].to(device),
                                                                    text_lengths=batch[1].to(device),
                                                                    speech=batch[2].to(device),
                                                                    speech_lengths=batch[3].to(device),
                                                                    step=step_counter,
                                                                    speaker_embeddings=batch[4].to(device),
                                                                    priot=batch[5].to(device),
                                                                    language_id=batch[6].to(device),
                                                                    return_mels=True,
                                                                    return_loss_dict=True)
                        pred_spemb = speaker_embedding_func.modules.embedding_model(predicted_mels,
                                                                                    torch.tensor([x / len(predicted_mels[0]) for x in batch[3]]))
                        gold_spemb = speaker_embedding_func.modules.embedding_model(batch[2].to(device),
                                                                                    torch.tensor([x / len(batch[2][0]) for x in batch[3]]))
                        # we have to recalculate the speaker embedding from our own mel because we project into a slightly different mel space
                        cosine_cycle_distance = torch.tensor(1.0) - F.cosine_similarity(pred_spemb.squeeze(), gold_spemb.squeeze(), dim=1).mean()
                        pairwise_cycle_distance = F.pairwise_distance(pred_spemb.squeeze(), gold_spemb.squeeze()).mean()
                        cycle_distance = cosine_cycle_distance + pairwise_cycle_distance
                        del pred_spemb
                        del predicted_mels
                        del gold_spemb
                        cycle_loss = cycle_distance * min(200, step_counter / 1200)
                        loss_dict["cycle"] = cycle_loss.item()
                        train_loss = train_loss + cycle_loss
                else:
                    if not use_speaker_embedding:
                        train_loss, loss_dict = net(text=batch[0].to(device),
                                                    text_lengths=batch[1].to(device),
                                                    speech=batch[2].to(device),
                                                    speech_lengths=batch[3].to(device),
                                                    prior=batch[4].to(device),
                                                    step=step_counter,
                                                    speaker_embeddings=None,
                                                    return_loss_dict=True)
                    else:
                        train_loss, predicted_mels, loss_dict = net(text=batch[0].to(device),
                                                                    text_lengths=batch[1].to(device),
                                                                    speech=batch[2].to(device),
                                                                    speech_lengths=batch[3].to(device),
                                                                    step=step_counter,
                                                                    speaker_embeddings=batch[4].to(device),
                                                                    prior=batch[5].to(device),
                                                                    return_mels=True,
                                                                    return_loss_dict=True)
                        pred_spemb = speaker_embedding_func.modules.embedding_model(predicted_mels,
                                                                                    torch.tensor([x / len(predicted_mels[0]) for x in batch[3]]))
                        gold_spemb = speaker_embedding_func.modules.embedding_model(batch[2].to(device),
                                                                                    torch.tensor([x / len(batch[2][0]) for x in batch[3]]))
                        # we have to recalculate the speaker embedding from our own mel because we project into a slightly different mel space
                        cycle_distance = torch.tensor(1.0) - F.cosine_similarity(pred_spemb.squeeze(), gold_spemb.squeeze(), dim=1).mean()
                        del pred_spemb
                        del predicted_mels
                        del gold_spemb
                        cycle_loss = cycle_distance * min(100, step_counter / 1200)
                        loss_dict["cycle"] = cycle_loss.item()
                        train_loss = train_loss + cycle_loss

                train_losses_this_epoch.append(train_loss.item())
                for loss_type in loss_dict:
                    if loss_type not in cumulative_loss_dict.keys():
                        cumulative_loss_dict[loss_type] = list()
                    cumulative_loss_dict[loss_type].append(loss_dict[loss_type])

            optimizer.zero_grad()
            scaler.scale(train_loss).backward()
            del train_loss
            step_counter += 1
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=False)
            scaler.step(optimizer)
            scaler.update()
            if freeze_encoder_until is not None and freeze_encoder_until < step_counter:
                for param in net.enc.parameters():
                    param.requires_grad = True
                freeze_encoder_until = None
                print("Encoder-weights are now unfrozen, good luck!")
            if freeze_decoder_until is not None and freeze_decoder_until < step_counter:
                for param in net.dec.parameters():
                    param.requires_grad = True
                freeze_decoder_until = None
                print("Decoder-weights are now unfrozen, good luck!")
            if freeze_embedding_until is not None and freeze_embedding_until < step_counter:
                for param in net.enc.embed.parameters():
                    param.requires_grad = True
                freeze_embedding_until = None
                print("Embedding-weights are now unfrozen, good luck!")
        net.eval()
        loss_this_epoch = sum(train_losses_this_epoch) / len(train_losses_this_epoch)
        if previous_error + collapse_margin < loss_this_epoch:
            print("Model Collapse detected! \nPrevious Loss: {}\nNew Loss: {}".format(previous_error, loss_this_epoch))
            print("Trying to reset to a stable state ...")
            path_to_checkpoint = get_most_recent_checkpoint(checkpoint_dir=save_directory)
            check_dict = torch.load(path_to_checkpoint, map_location=device)
            net.load_state_dict(check_dict["model"])
            if not fine_tune:
                optimizer.load_state_dict(check_dict["optimizer"])
                step_counter = check_dict["step_counter"]
                scaler.load_state_dict(check_dict["scaler"])
        else:
            previous_error = loss_this_epoch
            if epoch % epochs_per_save == 0:
                torch.save({
                    "model": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "step_counter": step_counter,
                }, os.path.join(save_directory, "checkpoint_{}.pt".format(step_counter)))
                delete_old_checkpoints(save_directory, keep=5)
                with torch.no_grad():
                    plot_attention(model=net,
                                   lang=lang,
                                   device=device,
                                   speaker_embedding=reference_speaker_embedding_for_att_plot,
                                   att_dir=save_directory,
                                   step=step_counter,
                                   language_id=reference_language_id)
                if step_counter > steps:
                    # DONE
                    return
            print("Epoch:        {}".format(epoch))
            print("Total Loss:   {}".format(round(loss_this_epoch, 3)))
            for loss_type in cumulative_loss_dict:
                print(f"    {loss_type}: {round(sum(cumulative_loss_dict[loss_type]) / len(cumulative_loss_dict[loss_type]), 3)}")
            print("Time elapsed: {} Minutes".format(round((time.time() - start_time) / 60)))
            print("Steps:        {}".format(step_counter))
        torch.cuda.empty_cache()
        net.train()
