import torch
import os
import json
import sys
import numpy as np
import pandas as pd
#sys.path.append("/home/behringe/hdd_behringe/IMS-Toucan") 
from Preprocessing.multilinguality.create_lang_emb_dataset import LangEmbDataset

class LangEmbPredictor(torch.nn.Module):
    def __init__(self, idim, odim, n_layers=3, dropout_rate=3, n_closest=5, use_phylo=True):
        super().__init__()
        self.linear = torch.nn.Linear(idim, odim)
        self.leaky_relu = torch.nn.LeakyReLU()
        self.batch_norm = torch.nn.BatchNorm1d(odim)
        self.layer_norm = torch.nn.LayerNorm(odim)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.layers = []
        for i in range(n_layers):
            self.layers.append(self.linear)
            self.layers.append(self.leaky_relu)
        self.layers = torch.nn.Sequential(*self.layers)

    def forward(self, xs):
        xs = self.layers(xs)
        return xs
    
class LangEmbPredictorLoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.criterion = torch.nn.MSELoss(reduction=reduction)
    
    def forward(self, outputs, targets):
        loss = self.criterion(outputs, targets)
        return loss