import os

import torch
from sentence_transformers import SentenceTransformer

from Utility.storage_config import MODELS_DIR

class SentenceEmbeddingExtractor():
    def __init__(self, cache_dir:str=os.path.join(MODELS_DIR, 'LM')):
        self.model = SentenceTransformer('dangvantuan/sentence-camembert-base', cache_folder=cache_dir)
        os.environ["TOKENIZERS_PARALLELISM"] = 'False'
    
    def encode(self, sentences: list[str]) -> torch.Tensor:
        return torch.as_tensor(self.model.encode(sentences))