from Utility.storage_config import PREPROCESSING_DIR
from transformers import pipeline
import torch
import os
import json
from tqdm import tqdm
import sys

def data(sentences):
    for sent in sentences:
        yield sent

if __name__ == '__main__':
    device = 'cuda:5'

    if not os.path.exists(os.path.join(PREPROCESSING_DIR, "Headlines", "emotion_sentences_full.pt")):
        with open("/mount/arbeitsdaten/synthesis/bottts/IMS-Toucan/Corpora/Headlines/gne-release-v1.0.jsonl") as f:
            sentences = [json.loads(line)["headline"] for line in f]

        print(f"Extracted {len(sentences)} sentences.")

        classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1, device=device)
        emotion_to_sents = {"anger":set(), "disgust":set(), "fear":set(), "joy":set(), "neutral":set(), "sadness":set(), "surprise":set()}
        
        for i, result in tqdm(enumerate(classifier(data(sentences), truncation=True, max_length=512, padding=True, batch_size=256)), total=len(sentences)):
            score = result[0]["score"]
            if score > 0.9:
                emotion = result[0]["label"]
                emotion_to_sents[emotion].add((sentences[i], score))
        for emotion, sents in emotion_to_sents.items():
            emotion_to_sents[emotion] = sorted(list(sents), key=lambda x: x[1], reverse=True)
        torch.save(emotion_to_sents, os.path.join(PREPROCESSING_DIR, "Headlines", "emotion_sentences_full.pt"))
    else:
        emotion_to_sents = torch.load(os.path.join(PREPROCESSING_DIR, "Headlines", "emotion_sentences_full.pt"), map_location='cpu')

    top_k = dict()
    for emotion, sents in emotion_to_sents.items():
        top_k[emotion] = [sent[0] for sent in sents[:20]]
    torch.save(top_k, os.path.join(PREPROCESSING_DIR, "Headlines", "emotion_sentences_top20.pt"))
