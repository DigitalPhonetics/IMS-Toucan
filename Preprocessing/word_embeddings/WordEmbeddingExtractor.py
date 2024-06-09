from abc import ABC, abstractmethod
import numpy as np
import os

from Utility.storage_config import MODELS_DIR

class WordEmbeddingExtractor(ABC):

    def __init__(self):
        self.cache_dir = os.path.join(MODELS_DIR, 'Language_Models')
        pass

    @abstractmethod
    def encode(self, sentences:list[str]) -> np.ndarray:
        pass