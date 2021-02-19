import torch
from torch.utils.data import Dataset


class MelGANDataset(Dataset):

    def __init__(self, list_of_paths, device=torch.device("cpu"), type="train"):
        if type == "train":
            self.list_of_paths = list_of_paths[:-100]
        elif type == "valid":
            self.list_of_paths = list_of_paths[-100:]
        else:
            print("unknown set type ('train' or 'valid' are allowed)")
        self.device = device

    def __getitem__(self, index):
        # load the audio from the path, clean it, process it into a spectrogram
        # return a pair of cleaned audio and spectrogram
        text = torch.LongTensor(self.feature_list[index][0]).to(self.device)
        text_len = torch.LongTensor([self.feature_list[index][1]]).to(self.device)
        speech = torch.transpose(torch.Tensor(self.feature_list[index][2]), 0, 1).to(self.device)
        speech_len = torch.LongTensor([self.feature_list[index][3]]).to(self.device)
        if self.spemb:
            spemb = torch.Tensor(self.feature_list[index][4]).to(self.device)
            return text, text_len, speech, speech_len, spemb
        return text, text_len, speech, speech_len

    def __len__(self):
        return len(self.list_of_paths)
