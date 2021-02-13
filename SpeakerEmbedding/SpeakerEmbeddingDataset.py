import json
import os
import random

import torch
from torch.utils.data import Dataset


class SpeakerEmbeddingDataset(Dataset):
    def __init__(self, path_to_feature_dump, size=250000, device="cuda"):
        speaker_to_melspec = dict()
        for el in os.listdir(path_to_feature_dump):
            with open(os.path.join(path_to_feature_dump, el), 'r') as fp:
                speaker_to_melspec.update(json.load(fp))
        self.len = size
        pure_pairs = list()
        impure_pairs = list()
        self.device = device
        speakers = list(speaker_to_melspec.keys())
        print("Data loaded. Preparing random sampling.")
        # collect some pure samples
        for _ in range(int(size / 2)):
            speaker = random.choice(speakers)
            specs = set()
            while len(specs) < 2:
                specs.add(random.choice(speaker_to_melspec[speaker]))
            pure_pairs.append(list(specs) + [[-1]])
        # collect some impure samples
        for _ in range(int(size / 2)):
            speaker_1 = random.choice(speakers)
            speaker_2 = random.choice(speakers)
            while speaker_2 == speaker_1:
                speaker_2 = random.choice(speakers)
            specs = list()
            specs.append(random.choice(speaker_to_melspec[speaker_1]))
            specs.append(random.choice(speaker_to_melspec[speaker_2]))
            impure_pairs.append(specs + [[1]])
        # combine the two
        self.datapoints = pure_pairs + impure_pairs

    def __getitem__(self, item):
        try:
            samp_1 = torch.Tensor(self.datapoints[item][0]).unsqueeze(0).unsqueeze(0).to(self.device)
            samp_2 = torch.Tensor(self.datapoints[item][1]).unsqueeze(0).unsqueeze(0).to(self.device)
            label = torch.Tensor(self.datapoints[item][2]).unsqueeze(0).unsqueeze(0).to(self.device)
            return samp_1, samp_2, label
        except IndexError:
            print("An element was queried that is not in the dataset: {}".format(item))

    def __len__(self):
        return self.len
