from Utility.storage_config import PREPROCESSING_DIR
from Preprocessing.sentence_embeddings.EmotionRoBERTaSentenceEmbeddingExtractor import EmotionRoBERTaSentenceEmbeddingExtractor
from Preprocessing.sentence_embeddings.BERTSentenceEmbeddingExtractor import BERTSentenceEmbeddingExtractor
from Preprocessing.sentence_embeddings.STSentenceEmbeddingExtractor import STSentenceEmbeddingExtractor
import torch
import os
from tqdm import tqdm

if __name__ == '__main__':
    device = 'cuda:6'
    emotion_prompts = torch.load(os.path.join(PREPROCESSING_DIR, "Yelp", "emotion_prompts_balanced_10000.pt"), map_location='cpu')
    sent_emb_extractor = STSentenceEmbeddingExtractor()
    emotion_prompts_sent_embs = {"anger":[], "disgust":[], "fear":[], "joy":[], "neutral":[], "sadness":[], "surprise":[]}
    for emotion in tqdm(list(emotion_prompts.keys())):
        for prompt in tqdm(emotion_prompts[emotion]):
            sent_emb = sent_emb_extractor.encode(sentences=[prompt]).squeeze()
            emotion_prompts_sent_embs[emotion].append(sent_emb)
    torch.save(emotion_prompts_sent_embs, os.path.join(PREPROCESSING_DIR, "Evaluation", "emotion_prompts_balanced_10000_sent_embs_stpara.pt"))
