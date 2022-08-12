"""
This script is meant to load a general embedding
checkpoint and then finetune it to 2 task specific
models: One for speakers and one for emotion.
"""
import os
import random

import soundfile
import torch
from tqdm import tqdm

from Preprocessing.AudioPreprocessor import AudioPreprocessor
from TrainingInterfaces.Spectrogram_to_Embedding.StyleEmbedding import StyleEmbedding
from Utility.diverse_losses import BarlowTwinsLoss
from Utility.diverse_losses import TripletLoss


class Dataset:

    def __init__(self):
        self.label_to_specs = dict()
        self.ap = AudioPreprocessor(input_sr=16000, output_sr=16000)

    def add_dataset(self, label_to_filelist):
        for label in label_to_filelist:
            for file in tqdm(label_to_filelist[label]):
                try:
                    wav, sr = soundfile.read(file)
                except RuntimeError:
                    print(f"bad file: {file}")
                    continue
                if sr != self.ap.sr:
                    self.ap = AudioPreprocessor(input_sr=sr, output_sr=16000)
                spec = self.ap.audio_to_mel_spec_tensor(wav, normalize=True)
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
                if os.path.isdir(os.path.join(root, speaker, emotion)):
                    for audio_file in os.listdir(os.path.join(root, speaker, emotion)):
                        label_to_filelist[emotion].append(os.path.join(root, speaker, emotion, audio_file))

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
    torch.save({"style_emb_func": finetuned_model.state_dict()}, "Models/Embedding/emotion_embedding_function.pt")


def finetune_model_speaker(gpu_id, resume_checkpoint, resume, finetune, model_dir):
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
    label_to_filelist = {"peter": ["test.wav"]}
    speaker_data.add_dataset(label_to_filelist)
    finetuned_model = finetune_model(speaker_data, device=device)
    torch.save({"style_emb_func": finetuned_model.state_dict()}, "Models/Embedding/speaker_embedding_function.pt")


def finetune_model(dataset, device, path_to_embed="Models/Embedding/embedding_function.pt"):
    # initialize losses
    contrastive_loss = TripletLoss(margin=1.0)
    non_contrastive_loss = BarlowTwinsLoss()

    # load model
    embed = StyleEmbedding()
    check_dict = torch.load(path_to_embed, map_location="cpu")
    embed.load_state_dict(check_dict["style_emb_func"])
    embed.to(device)

    # define optimizer
    optimizer = torch.optim.Adam(embed.parameters(), lr=0.001)
    optimizer.add_param_group({"params": non_contrastive_loss.parameters()})

    # train loop
    losses = list()
    for step in tqdm(range(50000)):
        for _ in range(32):  # effective batchsize through gradient accumulation. Just for more stable updates,
            # computationally slow, but simple to code, and it's still more than fast enough for a one-off script.
            anchor, positive, negative = dataset.sample_triplet()
            anchor_emb = embed(anchor, len(anchor))
            positive_emb = embed(positive, len(positive))
            negative_emb = embed(negative, len(negative))
            losses.append(contrastive_loss(anchor_emb, positive_emb, negative_emb) + (0.1 * non_contrastive_loss(anchor_emb, positive_emb)))
        loss = sum(losses) / len(losses)
        losses = list()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            print(loss.item())
    return embed
