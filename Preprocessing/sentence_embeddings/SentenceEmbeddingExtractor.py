from abc import ABC, abstractmethod
import torch
import os

from Utility.storage_config import MODELS_DIR

class SentenceEmbeddingExtractor(ABC):

    def __init__(self):
        self.cache_dir = os.path.join(MODELS_DIR, 'Language_Models')
        pass

    @abstractmethod
    def encode(self, sentences:list[str]) -> torch.Tensor:
        pass