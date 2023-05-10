import torch
from torch.nn import Linear
from torch.nn import Sequential
from torch.nn import Tanh

class SentenceEmbeddingAdaptor(torch.nn.Module):
    def __init__(self,
                sent_embed_dim=768,
                utt_embed_dim=64):
        super().__init__()

        self.adaptation_layers = Sequential(Linear(sent_embed_dim, sent_embed_dim // 2),
                                            Tanh(),
                                            Linear(sent_embed_dim // 2, sent_embed_dim // 4),
                                            Tanh(),
                                            Linear(sent_embed_dim // 4, utt_embed_dim))
        self.mse_criterion = torch.nn.MSELoss(reduction='mean')
    def forward(self,
                style_embedding=None,
                sentence_embedding=None,
                return_emb=False):
        if return_emb:
            sentence_embedding = torch.nn.functional.normalize(sentence_embedding)
            sentence_embedding = self.adaptation_layers(sentence_embedding)
            return sentence_embedding
        else:
            style_embedding = torch.nn.functional.normalize(style_embedding)
            sentence_embedding = torch.nn.functional.normalize(sentence_embedding)
            sentence_embedding = self.adaptation_layers(sentence_embedding)
            sent_style_loss = self.mse_criterion(sentence_embedding, style_embedding)
            return sent_style_loss
