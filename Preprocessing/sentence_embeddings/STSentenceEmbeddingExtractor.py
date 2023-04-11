import os

import torch
from sentence_transformers import SentenceTransformer

from Preprocessing.sentence_embeddings.SentenceEmbeddingExtractor import SentenceEmbeddingExtractor

class STSentenceEmbeddingExtractor(SentenceEmbeddingExtractor):
    def __init__(self, model:str='para', cache_dir:str=""):
        super().__init__()
        assert model in ['para', 'para_mini', 'distil', 'bloom', 'camembert']
        os.environ["TOKENIZERS_PARALLELISM"] = 'False'
        if cache_dir:
            self.cache_dir = cache_dir
        if model == 'para_mini':
            self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', cache_folder=self.cache_dir)
        if model == 'para':
            self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2', cache_folder=self.cache_dir)
        if model == 'distil':
            self.model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2', cache_folder=self.cache_dir)
        if model == 'bloom':
            self.model = SentenceTransformer('bigscience-data/sgpt-bloom-1b7-nli', cache_folder=self.cache_dir)
        if model == 'camembert':
            self.model = SentenceTransformer('dangvantuan/sentence-camembert-base', cache_folder=self.cache_dir)
    
    def encode(self, sentences: list[str]) -> torch.Tensor:
        return torch.as_tensor(self.model.encode(sentences))