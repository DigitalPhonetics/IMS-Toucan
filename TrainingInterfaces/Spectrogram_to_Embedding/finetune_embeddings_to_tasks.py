"""
This script is meant to load a general embedding
checkpoint and then finetune it to 2 task specific
models: One for speakers and one for emotion.
"""
import os
import random
import time

import soundfile
import torch
import wandb
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from Preprocessing.AudioPreprocessor import AudioPreprocessor
from TrainingInterfaces.Spectrogram_to_Embedding.StyleEmbedding import StyleEmbedding
from Utility.diverse_losses import BarlowTwinsLoss
from Utility.diverse_losses import TripletLoss
from Utility.storage_config import MODELS_DIR


class Dataset:

    def __init__(self):
        self.label_to_specs = dict()
        self.ap = AudioPreprocessor(input_sr=16000, output_sr=16000)

    def add_dataset(self, label_to_filelist):
        for label in tqdm(label_to_filelist):
            for file in label_to_filelist[label]:
                try:
                    wav, sr = soundfile.read(file)
                except RuntimeError:
                    print(f"bad file: {file}")
                    continue
                if sr != self.ap.sr:
                    self.ap = AudioPreprocessor(input_sr=sr, output_sr=16000)
                spec = self.ap.audio_to_mel_spec_tensor(wav, normalize=True).transpose(0, 1)
                if label not in self.label_to_specs:
                    self.label_to_specs[label] = list()
                self.label_to_specs[label].append(spec)

    def sample_triplet(self):
        """
        returns two spectrograms with the same label and one spectrogram with a different label w.r.t. the current task
        """
        label = random.choice(list(self.label_to_specs.keys()))
        neg_label = random.choice(list(self.label_to_specs.keys()))
        while neg_label == label:
            neg_label = random.choice(list(self.label_to_specs.keys()))
        return random.choice(self.label_to_specs[label]), random.choice(self.label_to_specs[label]), random.choice(self.label_to_specs[neg_label])


def finetune_model_emotion(gpu_id, resume_checkpoint, resume, finetune, model_dir):
    """
    finetune model on data with different emotion categories
    arguments are there for compatibility, but unused.
    """
    if gpu_id == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")

    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
        device = torch.device("cuda")

    torch.manual_seed(131714)
    random.seed(131714)
    torch.random.manual_seed(131714)
    emo_data = Dataset()
    label_to_filelist = dict()

    # add aesdd
    root = "/mount/resources/speech/corpora/ActedEmotionalSpeechDynamicDatabase"
    for emotion in os.listdir(root):
        if emotion != "Tools and Documentation":
            if emotion not in label_to_filelist:
                label_to_filelist[emotion] = list()
            for audio_file in os.listdir(os.path.join(root, emotion)):
                label_to_filelist[emotion].append(os.path.join(root, emotion, audio_file))

    # add ADEPT
    root = "/mount/resources/speech/corpora/ADEPT/wav_44khz/emotion"
    for emotion in os.listdir(root):
        if emotion != "Tools and Documentation":
            if emotion not in label_to_filelist:
                label_to_filelist[emotion] = list()
            for audio_file in os.listdir(os.path.join(root, emotion)):
                label_to_filelist[emotion].append(os.path.join(root, emotion, audio_file))

    # add ESDS
    root = "/mount/resources/speech/corpora/Emotional_Speech_Dataset_Singapore"
    for speaker in os.listdir(root):
        if os.path.isdir(os.path.join(root, speaker)):
            for emo in os.listdir(os.path.join(root, speaker)):
                if emo == "Sad":
                    emotion = "sadness"
                elif emo == "Neutral":
                    emotion = "neutral"
                elif emo == "Happy":
                    emotion = "joy"
                elif emo == "Angry":
                    emotion = "anger"
                elif emo == "Surprise":
                    emotion = "surprised"
                else:
                    continue
                if emotion not in label_to_filelist:
                    label_to_filelist[emotion] = list()
                if os.path.isdir(os.path.join(root, speaker, emo)):
                    for audio_file in os.listdir(os.path.join(root, speaker, emo)):
                        label_to_filelist[emotion].append(os.path.join(root, speaker, emo, audio_file))

    # add RAVDESS
    root = "/mount/resources/speech/corpora/RAVDESS"
    for speaker in os.listdir(root):
        if os.path.isdir(os.path.join(root, speaker)):
            for audio_file in os.listdir(os.path.join(root, speaker)):
                if audio_file.split("-")[1] == "01":
                    if audio_file.split("-")[2] == "01":
                        emotion = "neutral"
                    elif audio_file.split("-")[2] == "03":
                        emotion = "joy"
                    elif audio_file.split("-")[2] == "04":
                        emotion = "sadness"
                    elif audio_file.split("-")[2] == "05":
                        emotion = "anger"
                    elif audio_file.split("-")[2] == "06":
                        emotion = "fear"
                    elif audio_file.split("-")[2] == "07":
                        emotion = "disgust"
                    elif audio_file.split("-")[2] == "08":
                        emotion = "surprised"
                    else:
                        continue
                    if emotion not in label_to_filelist:
                        label_to_filelist[emotion] = list()
                    label_to_filelist[emotion].append(os.path.join(root, speaker, audio_file))
    print(label_to_filelist.keys())

    emo_data.add_dataset(label_to_filelist)
    finetuned_model = finetune_model(emo_data, device=device)
    torch.save({"style_emb_func": finetuned_model.state_dict()}, os.path.join(MODELS_DIR, "Embedding", "emotion_embedding_function.pt"))


def finetune_model_speaker(gpu_id, resume_checkpoint, resume, finetune, model_dir, use_wandb, wandb_resume_id):
    """
    finetune model on data with different speakers
    arguments are there for compatibility, but unused.

    """
    if gpu_id == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")

    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
        device = torch.device("cuda")

    torch.manual_seed(131714)
    random.seed(131714)
    torch.random.manual_seed(131714)
    speaker_data = Dataset()
    label_to_filelist = dict()

    # add hui_others
    root = "/mount/resources/speech/corpora/HUI_German/others"
    for speaker in os.listdir(root):
        label_to_filelist[speaker] = list()
        spk_root = f"{root}/{speaker}"
        for el in os.listdir(spk_root):
            if os.path.isdir(os.path.join(spk_root, el)):
                with open(os.path.join(spk_root, el, "metadata.csv"), "r", encoding="utf8") as file:
                    lookup = file.read()
                for line in lookup.split("\n"):
                    if line.strip() != "":
                        wav_path = os.path.join(spk_root, el, "wavs", line.split("|")[0] + ".wav")
                        if os.path.exists(wav_path):
                            if len(label_to_filelist[speaker]) > 15:
                                break
                            label_to_filelist[speaker].append(wav_path)

    # add a little of Nancy
    label_to_filelist["Nancy"] = list()
    root = "/mount/resources/speech/corpora/NancyKrebs"
    with open(os.path.join(root, "metadata.csv"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n")[:100]:
        if line.strip() != "":
            wav_path = os.path.join(root, "wav", line.split("|")[0] + ".wav")
            if os.path.exists(wav_path):
                label_to_filelist["Nancy"].append(wav_path)

    # add LibriTTS
    path_train = "/mount/resources/speech/corpora/LibriTTS/all_clean"
    for speaker in os.listdir(path_train):
        label_to_filelist[speaker] = list()
        for chapter in os.listdir(os.path.join(path_train, speaker)):
            for file in os.listdir(os.path.join(path_train, speaker, chapter)):
                if file.endswith("normalized.txt"):
                    wav_file = file.split(".")[0] + ".wav"
                    if len(label_to_filelist[speaker]) > 15:
                        break
                    label_to_filelist[speaker].append(os.path.join(path_train, speaker, chapter, wav_file))

    # add ESDS
    root = "/mount/resources/speech/corpora/Emotional_Speech_Dataset_Singapore"
    for speaker in os.listdir(root):
        label_to_filelist[speaker] = list()
        if os.path.isdir(os.path.join(root, speaker)):
            for emo in os.listdir(os.path.join(root, speaker)):
                if os.path.isdir(os.path.join(root, speaker, emo)):
                    for audio_file in os.listdir(os.path.join(root, speaker, emo)):
                        label_to_filelist[speaker].append(os.path.join(root, speaker, emo, audio_file))

    # add RAVDESS
    root = "/mount/resources/speech/corpora/RAVDESS"
    for speaker in os.listdir(root):
        label_to_filelist[speaker] = list()
        if os.path.isdir(os.path.join(root, speaker)):
            for audio_file in os.listdir(os.path.join(root, speaker)):
                label_to_filelist[speaker].append(os.path.join(root, speaker, audio_file))

    # add MLS it
    lang = "italian"
    root = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train/audio"
    path_train = root
    for speaker in os.listdir(path_train):
        label_to_filelist[speaker] = list()
        for chapter in os.listdir(os.path.join(path_train, speaker)):
            for file in os.listdir(os.path.join(path_train, speaker, chapter)):
                if len(label_to_filelist[speaker]) > 15:
                    break
                label_to_filelist[speaker].append(os.path.join(path_train, speaker, chapter, file))

    # add MLS fr
    lang = "french"
    root = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train/audio"
    path_train = root
    for speaker in os.listdir(path_train):
        label_to_filelist[speaker] = list()
        for chapter in os.listdir(os.path.join(path_train, speaker)):
            for file in os.listdir(os.path.join(path_train, speaker, chapter)):
                if len(label_to_filelist[speaker]) > 15:
                    break
                label_to_filelist[speaker].append(os.path.join(path_train, speaker, chapter, file))

    # add MLS dt
    lang = "dutch"
    root = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train/audio"
    path_train = root
    for speaker in os.listdir(path_train):
        label_to_filelist[speaker] = list()
        for chapter in os.listdir(os.path.join(path_train, speaker)):
            for file in os.listdir(os.path.join(path_train, speaker, chapter)):
                if len(label_to_filelist[speaker]) > 15:
                    break
                label_to_filelist[speaker].append(os.path.join(path_train, speaker, chapter, file))

    # add MLS pl
    lang = "polish"
    root = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train/audio"
    path_train = root
    for speaker in os.listdir(path_train):
        label_to_filelist[speaker] = list()
        for chapter in os.listdir(os.path.join(path_train, speaker)):
            for file in os.listdir(os.path.join(path_train, speaker, chapter)):
                if len(label_to_filelist[speaker]) > 15:
                    break
                label_to_filelist[speaker].append(os.path.join(path_train, speaker, chapter, file))

    # add MLS sp
    lang = "spanish"
    root = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train/audio"
    path_train = root
    for speaker in os.listdir(path_train):
        label_to_filelist[speaker] = list()
        for chapter in os.listdir(os.path.join(path_train, speaker)):
            for file in os.listdir(os.path.join(path_train, speaker, chapter)):
                if len(label_to_filelist[speaker]) > 15:
                    break
                label_to_filelist[speaker].append(os.path.join(path_train, speaker, chapter, file))

    # add MLS pt
    lang = "portuguese"
    root = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train/audio"
    path_train = root
    for speaker in os.listdir(path_train):
        label_to_filelist[speaker] = list()
        for chapter in os.listdir(os.path.join(path_train, speaker)):
            for file in os.listdir(os.path.join(path_train, speaker, chapter)):
                if len(label_to_filelist[speaker]) > 15:
                    break
                label_to_filelist[speaker].append(os.path.join(path_train, speaker, chapter, file))

    speaker_data.add_dataset(label_to_filelist)
    finetuned_model = finetune_model(speaker_data, device=device)
    torch.save({"style_emb_func": finetuned_model.state_dict()}, os.path.join(MODELS_DIR, "Embedding", "speaker_embedding_function.pt"))


def finetune_model(dataset, device, path_to_embed=os.path.join(MODELS_DIR, "Embedding", "embedding_function.pt")):
    # initialize losses

    wandb.init(name=f"{__name__.split('.')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}", id=None, resume=None)

    contrastive_loss = TripletLoss(margin=1.0)
    non_contrastive_loss = BarlowTwinsLoss(vector_dimensions=64).to(device)

    # load model
    embed = StyleEmbedding()
    check_dict = torch.load(path_to_embed, map_location="cpu")
    embed.load_state_dict(check_dict["style_emb_func"])
    embed.to(device)

    # define optimizer
    optimizer = torch.optim.AdamW(embed.parameters(), lr=0.001)
    optimizer.add_param_group({"params": non_contrastive_loss.parameters()})

    con_losses = list()
    non_con_losses = list()

    # train loop
    for step in tqdm(range(10000)):
        anchors = list()
        anchor_lens = list()
        positives = list()
        positive_lens = list()
        negatives = list()
        negative_lens = list()

        # build a batch (I know this is slower than using a torch dataset, but it's sufficient and a bit quicker to implement.)
        for _ in range(128):
            anchor, positive, negative = dataset.sample_triplet()
            anchors.append(anchor)
            anchor_lens.append(torch.LongTensor([len(anchor)]))
            positives.append(positive)
            positive_lens.append(torch.LongTensor([len(positive)]))
            negatives.append(negative)
            negative_lens.append(torch.LongTensor([len(negative)]))

        # collate and pad
        anchor = pad_sequence(anchors, batch_first=True)
        anchor_len = torch.stack(anchor_lens)
        positive = pad_sequence(positives, batch_first=True)
        positive_len = torch.stack(positive_lens)
        negative = pad_sequence(negatives, batch_first=True)
        negative_len = torch.stack(negative_lens)

        # embed
        anchor_emb = embed(anchor.to(device), anchor_len.to(device))
        positive_emb = embed(positive.to(device), positive_len.to(device))
        negative_emb = embed(negative.to(device), negative_len.to(device))

        # calculate loss on embeddings and update
        con_loss = contrastive_loss(anchor_emb, positive_emb, negative_emb)
        con_losses.append(con_loss.item())
        if step % 10 == 0 and step < 5000:
            non_con_loss = non_contrastive_loss(anchor_emb, positive_emb)
            non_con_losses.append(non_con_loss.item())
        else:
            non_con_loss = 0.0
        loss = con_loss + non_con_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log
        if step % 100 == 0:
            wandb.log({
                "contrastive": sum(con_losses) / len(con_losses),
                "triplet"    : sum(non_con_losses) / len(non_con_losses),
                "Steps"      : step
            })

            print(f"\nStep: {step}")
            print(f"Contrastive:     {sum(con_losses) / len(con_losses)}")
            print(f"Non-Contrastive: {sum(non_con_losses) / len(non_con_losses)}")
            con_losses = list()
            non_con_losses = list()

    return embed
