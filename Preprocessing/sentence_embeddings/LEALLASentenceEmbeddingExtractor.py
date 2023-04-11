import os
import tensorflow_hub as hub
import tensorflow_text as text # required even if import is not used here
import tensorflow as tf
import torch

from Preprocessing.sentence_embeddings.SentenceEmbeddingExtractor import SentenceEmbeddingExtractor

class LEALLASentenceEmbeddingExtractor(SentenceEmbeddingExtractor):
    def __init__(self, cache_dir:str=""):
        super().__init__()
        if cache_dir:
            self.cache_dir = cache_dir
        os.environ['TFHUB_CACHE_DIR']=self.cache_dir
        self.model = hub.KerasLayer("https://tfhub.dev/google/LEALLA/LEALLA-base/1")

    def encode(self, sentences: list[str]) -> torch.Tensor:
        return torch.as_tensor(self.model(tf.constant(sentences)).numpy())