from Utility.storage_config import PREPROCESSING_DIR
from datasets import load_dataset
from transformers import pipeline
import random
import torch
import os
from tqdm import tqdm
import nltk
nltk.download('punkt')

def data(sentences):
    for sent in sentences:
        yield sent

if __name__ == '__main__':
    device = 'cuda:5'
    yelp = load_dataset("yelp_review_full", split="train", cache_dir=os.path.join(PREPROCESSING_DIR, 'Yelp'))
    yelp_sents = []
    for review in tqdm(yelp[:]["text"]):
        sentences = nltk.sent_tokenize(review)
        for sent in sentences:  
            if len(sent.split()) < 50:
                yelp_sents.append(sent)

    print(f"Extracted {len(yelp_sents)} sentences.")

    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1, device=device)
    emotion_to_prompts = {"anger":[], "disgust":[], "fear":[], "joy":[], "neutral":[], "sadness":[], "surprise":[]}

    for i, result in tqdm(enumerate(classifier(data(yelp_sents), truncation=True, max_length=512, padding=True, batch_size=256)), total=len(yelp_sents)):
        score = result[0]["score"]
        if score > 0.8:
            emotion = result[0]["label"]
            emotion_to_prompts[emotion].append(yelp_sents[i])
    torch.save(emotion_to_prompts, os.path.join(PREPROCESSING_DIR, "Yelp", "emotion_prompts_full.pt"))

    emotion_to_prompts_balanced = dict()
    for emotion, prompts in tqdm(emotion_to_prompts.items()):
        emotion_to_prompts_balanced[emotion] = random.sample(prompts, 10000)
    
    torch.save(emotion_to_prompts, os.path.join(PREPROCESSING_DIR, "Yelp", "emotion_prompts_balanced_10000.pt"))
