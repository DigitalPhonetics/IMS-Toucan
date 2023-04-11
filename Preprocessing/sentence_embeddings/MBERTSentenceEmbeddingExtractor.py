import torch
from transformers import BertTokenizer, BertModel

from Preprocessing.sentence_embeddings.SentenceEmbeddingExtractor import SentenceEmbeddingExtractor

class MBERTSentenceEmbeddingExtractor(SentenceEmbeddingExtractor):
    def __init__(self, cache_dir:str="", pooling:str='second_to_last_mean', device=torch.device("cuda")):
        super().__init__()
        if cache_dir:
            self.cache_dir = cache_dir
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', cache_dir=self.cache_dir)
        self.model = BertModel.from_pretrained("bert-base-multilingual-cased", cache_dir=self.cache_dir).to(device)
        assert pooling in ["cls", "last_mean", "second_to_last_mean"]
        self.pooling = pooling
        self.device = device
    
    def encode(self, sentences: list[str]) -> torch.Tensor:
        if self.pooling == "cls":
            encoded_input = self.tokenizer(sentences, padding=True, return_tensors='pt').to(self.device)
            return self.model(**encoded_input).last_hidden_state[:,0].detach().cpu()
        if self.pooling == "last_mean":
            encoded_input = self.tokenizer(sentences, padding=True, return_tensors='pt').to(self.device)
            token_embeddings = self.model(**encoded_input, output_hidden_states=True).last_hidden_state
            return torch.mean(token_embeddings, dim=1).detach().cpu()
        if self.pooling == "second_to_last_mean":
            encoded_input = self.tokenizer(sentences, padding=True, return_tensors='pt').to(self.device)
            token_embeddings = self.model(**encoded_input, output_hidden_states=True).hidden_states[-2]
            return torch.mean(token_embeddings, dim=1).detach().cpu()