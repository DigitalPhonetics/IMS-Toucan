import torch

from Preprocessing.sentence_embeddings.SentenceEmbeddingExtractor import SentenceEmbeddingExtractor
from Preprocessing.sentence_embeddings.laserembeddings.laser import Laser

class LASERSentenceEmbeddingExtractor(SentenceEmbeddingExtractor):
    def __init__(self):
        super().__init__()
        self.model = Laser(mode='spm')

    def encode(self, sentences: list[str]) -> torch.Tensor:
        return torch.as_tensor(self.model.embed_sentences(sentences))