from Utility.storage_config import PREPROCESSING_DIR
import torch
import os
from datasets import load_dataset

if __name__ == '__main__':
    device = 'cuda:5'

    dataset = load_dataset("daily_dialog", split="train", cache_dir=os.path.join(PREPROCESSING_DIR, 'DailyDialogues'))
    id_to_emotion = {0: "neutral", 1: "anger", 2: "disgust", 3: "fear", 4: "joy", 5: "sadness", 6: "surprise"}
    emotion_to_sents = emotion_to_sents = {"anger":[], "disgust":[], "fear":[], "joy":[], "neutral":[], "sadness":[], "surprise":[]}

    for dialog, emotions in zip(dataset["dialog"], dataset["emotion"]):
        for sent, emotion in zip(dialog, emotions):
            emotion_to_sents[id_to_emotion[emotion]].append(sent.strip())

    torch.save(emotion_to_sents, os.path.join(PREPROCESSING_DIR, "DailyDialogues", "emotion_sentences."))
