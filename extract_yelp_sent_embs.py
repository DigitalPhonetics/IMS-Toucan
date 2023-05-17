from Utility.storage_config import PREPROCESSING_DIR
from Preprocessing.sentence_embeddings.EmotionRoBERTaSentenceEmbeddingExtractor import EmotionRoBERTaSentenceEmbeddingExtractor as SentenceEmbeddingExtractor
import torch
import os
from tqdm import tqdm

if __name__ == '__main__':
    device = 'cuda:0'
    emotion_prompts = torch.load(os.path.join(PREPROCESSING_DIR, "Yelp", "emotion_prompts_large.pt"), map_location='cpu')
    sent_emb_extractor = SentenceEmbeddingExtractor(pooling='cls', device=device)
    emotion_prompts_sent_embs = {"anger":[], "disgust":[], "fear":[], "joy":[], "neutral":[], "sadness":[], "surprise":[]}
    for emotion in tqdm(list(emotion_prompts.keys())):
        for prompt in tqdm(emotion_prompts[emotion]):
            sent_emb = sent_emb_extractor.encode(sentences=[prompt]).squeeze()
            emotion_prompts_sent_embs[emotion].append(sent_emb)
    torch.save(emotion_prompts_sent_embs, os.path.join(PREPROCESSING_DIR, "Yelp", "emotion_prompts_large_sent_embs_emoBERT.pt"))
