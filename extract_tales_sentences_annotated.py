from Utility.storage_config import PREPROCESSING_DIR
import torch
import os
from tqdm import tqdm
import pandas as pd
import csv

if __name__ == '__main__':
    device = 'cuda:5'
    data_dir = "/mount/arbeitsdaten/synthesis/bottts/IMS-Toucan/Corpora/Tales"
    id_to_emotion = {"N": "neutral", "A": "anger", "D": "disgust", "F": "fear", "H": "joy", "Sa": "sadness", "Su+": "surprise", "Su-": "surprise"}
    emotion_to_sents = emotion_to_sents = {"anger":[], "disgust":[], "fear":[], "joy":[], "neutral":[], "sadness":[], "surprise":[]}
    
    for author in tqdm(os.listdir(data_dir)):
        if not author.endswith(".pt"):
            for file in os.listdir(os.path.join(data_dir, author, "emmood")):
                df = pd.read_csv(os.path.join(data_dir, author, "emmood", file), sep="\t", header=None, quoting=csv.QUOTE_NONE)
                for index, (sent_id, emo, mood, sent) in df.iterrows():
                    emotions = emo.split(":")
                    if emotions[0] == emotions[1]:
                        emotion_to_sents[id_to_emotion[emotions[0]]].append(sent)

    torch.save(emotion_to_sents, os.path.join(PREPROCESSING_DIR, "Tales", "emotion_sentences.pt"))
