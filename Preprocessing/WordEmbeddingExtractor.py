import os

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import CamembertModel
from transformers import CamembertTokenizerFast

from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
from Preprocessing.TextFrontend import remove_french_spacing
from Utility.storage_config import MODELS_DIR


class WordEmbeddingExtractor():
    def __init__(self, cache_dir: str = os.path.join(MODELS_DIR, 'LM'), device=torch.device("cuda")):
        self.tokenizer = CamembertTokenizerFast.from_pretrained('camembert-base', cache_dir=cache_dir)
        self.model = CamembertModel.from_pretrained("camembert-base", cache_dir=cache_dir).to(device)
        self.model.eval()
        self.device = device
        self.tf = ArticulatoryCombinedTextFrontend(language="fr_no_flair")
        # phonemizer reduces or expands some words
        # we try to account for this here to get matching sequence lengths
        # more case might have to be added in the future
        self.replacements = [("parce que", "parce-que"), 
                             ("parce qu'il", "parce-qu'il"), 
                             ("parce qu'ils", "parce-qu'ils") ,
                             ("temps en temps", "temps-en-temps"), 
                             ("sud est", "sud-est"),
                             ("tout le monde", "tout-le-monde"), 
                             ("qu'est-ce que", "qu'est-ce-que"),
                             ("ceux-ci", "ceux ci"),
                             ("celles-ci", "celles ci"),
                             ("celle-ci", "celle ci"),
                             ("celle-là", "celle là"),
                             ("tant mieux", "tant-mieux"),
                             ("ceux-là", "ceux là"),
                             ("leAsseyez", "le Asseyez"),
                             ("parce qu", "parce-qu"),
                             ("tout le temps", "tout-le-temps"),
                             ("tant pis", "tant-pis"),
                             ("celles-là", "celles là"),
                             ("as tu", "as-tu"),
                             ("l'ai je", "l'ai-je"),
                             ("j'ai je", "j'ai-je"),
                             ("nord est", "nord-est"),
                             ("ta ent", "ta-ent"),
                             ("n'est ce", "n'est-ce"),
                             ("of the", "of-the"),
                             ("déveiapp ent", "déveiapp-ent"),
                             ("jug ent", "jug-ent"),
                             (" ent ", "-ent "),
                             ("vip", "v i p"),
                             ("qu'est ce que", "qu'est-ce-que"),
                             ("ai je", "ai-je"),
                             ("jureTu", "jure Tu"),
                             ("tellementAh", "tellement Ah"),
                             ("votreTe", "votre Te"),
                             ("desDes", "des Des"),
                             ("PaoShen", "Pao Shen"),
                             ("textesOh", "textes Oh"),
                             ("prendreLa", "prendre La"),
                             ("queBien", "que Bien"),
                             # and upper case
                             ("Parce que", "Parce-que"),
                             ("Parce qu'il", "Parce-qu'il"),
                             ("Parce qu'ils", "Parce-qu'ils"),
                             ("Temps en temps", "Temps-en-temps"),
                             ("Sud est", "Sud-est"),
                             ("Tout le monde", "Tout-le-monde"),
                             ("Qu'est-ce que", "Qu'est-ce-que"),
                             ("Ceux-ci", "Ceux ci"),
                             ("Celles-ci", "Celles ci"),
                             ("Celle-ci", "Celle ci"),
                             ("Celle-là", "Celle là"),
                             ("Tant mieux", "Tant-mieux"),
                             ("Ceux-là", "Ceux là"),
                             ("Parce qu", "Parce-qu"),
                             ("Tout le temps", "Tout-le-temps"),
                             ("Tant pis", "Tant-pis"),
                             ("Celles-là", "Celles là"),
                             ("As tu", "As-tu"),
                             ("L'ai je", "L'ai-je"),
                             ("J'ai je", "J'ai-je"),
                             ("Nord est", "Nord est"),
                             ("Ta ent", "Ta-ent"),
                             ("N'est ce", "N'est-ce"),
                             ("Of the", "Of-the"),
                             ("Déveiapp ent", "Déveiapp ent"),
                             ("Jug ent", "Jug-ent"),
                             ("VIP", "V I P"),
                             ("Qu'est ce que", "Qu'est-ce-que"),
                             ("Ai je", "Ai-je"),
                             ]

    def encode(self, sentences: list[str]) -> np.ndarray:
        if type(sentences) == str:
            sentences = [sentences]
        # apply spacing
        sentences = [remove_french_spacing(sent) for sent in sentences]
        sentences_replaced = []
        # replace words
        for sent in sentences:
            phone_string = self.tf.get_phone_string(sent)
            for replacement in self.replacements:
                sent = sent.replace(replacement[0], replacement[1])
            if len(phone_string.split()) != len(sent.split()):
                print("Warning: unhandled length mismatch in following sentence, consider modifying replacements in word embedding extractor.")
                print(sent)
                print(phone_string)
            sentences_replaced.append(sent)
        sentences = sentences_replaced
        # tokenize and encode sentences
        encoded_input = self.tokenizer(sentences, padding=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            # get all hidden states
            hidden_states = self.model(**encoded_input, output_hidden_states=True).hidden_states
        # stack and sum last 4 layers for each token
        token_embeddings = torch.stack([hidden_states[-4], hidden_states[-3], hidden_states[-2], hidden_states[-1]]).sum(0).squeeze()
        if len(sentences) == 1:
            token_embeddings = token_embeddings.unsqueeze(0)
        word_embeddings_list = []
        lens = []
        for batch_id in range(len(sentences)):
            # get word ids corresponding to token embeddings
            word_ids = encoded_input.word_ids(batch_id)
            word_ids_set = set([word_id for word_id in word_ids if word_id is not None])
            # get ids of hidden states of sub tokens for each word
            token_ids_words = [[t_id for t_id, word_id in enumerate(word_ids) if word_id == w_id] for w_id in word_ids_set]
            # combine hidden states of sub tokens for each word
            word_embeddings = torch.stack([token_embeddings[batch_id, token_ids_word].mean(dim=0) for token_ids_word in token_ids_words])
            word_embeddings_list.append(word_embeddings)
            # save sentence lengths
            lens.append(word_embeddings.shape[0])
        # pad tensors to max sentence lenth of batch
        word_embeddings_batch = pad_sequence(word_embeddings_list, batch_first=True).detach()
        # return word embeddings for each word in each sentence along with sentence lengths
        return word_embeddings_batch, lens
