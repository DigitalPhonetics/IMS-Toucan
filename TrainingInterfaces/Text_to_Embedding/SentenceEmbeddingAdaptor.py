import torch
from torch.nn import Linear
from torch.nn import Sequential
from torch.nn import Tanh

class SentenceEmbeddingAdaptor(torch.nn.Module):
    def __init__(self,
                sent_embed_dim=768,
                utt_embed_dim=64,
                speaker_embed_dim=None):
        super().__init__()

        self.use_speaker_embedding = speaker_embed_dim is not None
        self.speaker_embed_dim = speaker_embed_dim

        if self.use_speaker_embedding:
            self.adaptation_layers = Sequential(Linear(sent_embed_dim + speaker_embed_dim, 632),
                                                Tanh(),
                                                Linear(632, 324),
                                                Tanh(),
                                                Linear(324, 162),
                                                Tanh(),
                                                Linear(162, utt_embed_dim))
        else:
            self.adaptation_layers = Sequential(Linear(sent_embed_dim, sent_embed_dim // 2),
                                                Tanh(),
                                                Linear(sent_embed_dim // 2, sent_embed_dim // 4),
                                                Tanh(),
                                                Linear(sent_embed_dim // 4, utt_embed_dim))
        self.mse_criterion = torch.nn.MSELoss(reduction='mean')
    def forward(self,
                style_embedding=None,
                sentence_embedding=None,
                speaker_embedding=None,
                return_emb=False):
        if return_emb:
            sentence_embedding = torch.nn.functional.normalize(sentence_embedding)
            if self.use_speaker_embedding:
                speaker_embedding = torch.nn.functional.normalize(speaker_embedding)
                sentence_embedding = torch.cat([sentence_embedding, speaker_embedding], dim=1)
            sentence_embedding = self.adaptation_layers(sentence_embedding)
            return sentence_embedding
        else:
            style_embedding = torch.nn.functional.normalize(style_embedding)
            sentence_embedding = torch.nn.functional.normalize(sentence_embedding)
            if self.use_speaker_embedding:
                speaker_embedding = torch.nn.functional.normalize(speaker_embedding)
                sentence_embedding = torch.cat([sentence_embedding, speaker_embedding], dim=1)
            sentence_embedding = self.adaptation_layers(sentence_embedding)
            sent_style_loss = self.mse_criterion(sentence_embedding, style_embedding)
            return sent_style_loss
