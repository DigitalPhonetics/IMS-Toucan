from Utility.storage_config import PREPROCESSING_DIR
from datasets import load_dataset
from transformers import pipeline
import re
import torch
import os
from tqdm import tqdm

def data(sentences):
    for sent in sentences:
        yield sent

if __name__ == '__main__':
    device = 'cuda:0'
    yelp = load_dataset("yelp_review_full", split="train", cache_dir=os.path.join(PREPROCESSING_DIR, 'Yelp'))
    yelp_sents = []
    for sent in yelp[:]["text"]:
        sent = re.split('([.?!])', sent)
        try:
            sent = sent[0] + sent[1]
        except IndexError:
            continue
        if len(sent.split()) < 50:
            yelp_sents.append(sent)

    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1, device=device)
    #emotion_to_id = {"anger":0, "disgust":1, "fear":2, "joy":3, "neutral":4, "sadness":5, "surprise":6}
    emotion_to_prompts = {"anger":[], "disgust":[], "fear":[], "joy":[], "neutral":[], "sadness":[], "surprise":[]}

    '''
    for sent in tqdm(yelp_sents):
        try:
            score = classifier(sent)[0]
            for emotion in list(emotion_to_id.keys()):
                if score[emotion_to_id[emotion]]["score"] > 0.8:
                    emotion_to_prompts[emotion].append(sent)
        except RuntimeError:
            print("sentence could not be classified")
    '''
    for i, result in tqdm(enumerate(classifier(data(yelp_sents), truncation=True, max_length=512, padding=True)), total=len(yelp_sents)):
        score = result[0]["score"]
        if score > 0.8:
            emotion = result[0]["label"]
            emotion_to_prompts[emotion].append(yelp_sents[i])
    torch.save(emotion_to_prompts, os.path.join(PREPROCESSING_DIR, "Yelp", "emotion_prompts_large.pt"))
