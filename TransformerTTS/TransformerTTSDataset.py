import json

import torch
from torch.utils.data import Dataset


class TransformerTTSDataset(Dataset):

    def __init__(self, path_to_feature_dump, device="cuda", spemb=False):
        with open(path_to_feature_dump, 'r') as fp:
            self.feature_list = json.load(fp)
        self.spemb = spemb
        self.device = device

    def __getitem__(self, index):
        # create tensors on correct device
        text = torch.Tensor(self.feature_list[index][0]).to(self.device)
        text_len = torch.Tensor(self.feature_list[index][1]).to(self.device)
        speech = torch.Tensor(self.feature_list[index][2]).to(self.device)
        speech_len = torch.Tensor(self.feature_list[index][3]).to(self.device)
        if self.spemb:
            spemb = torch.Tensor(self.feature_list[index][4]).to(self.device)
            return text, text_len, speech, speech_len, spemb
        return text, text_len, speech, speech_len

    def __len__(self):
        return len(self.feature_list)
