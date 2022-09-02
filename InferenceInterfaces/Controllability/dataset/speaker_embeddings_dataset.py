import os

import numpy as np
import torch


class SpeakerEmbeddingsDataset(torch.utils.data.Dataset):

    def __init__(self, feature_path, device, mode='utterance'):
        super(SpeakerEmbeddingsDataset, self).__init__()

        modes = ['utterance', 'speaker']
        assert mode in modes, f'mode: {mode} is not supported'
        if mode == 'utterance':
            self.mode = 'utt'
        elif mode == 'speaker':
            self.mode = 'spk'

        self.device = device

        self.x, self.speakers = self._load_features(feature_path)
        # unique_speakers = set(self.speakers)
        # spk2class = dict(zip(unique_speakers, range(len(unique_speakers))))
        # #self.x = self._reformat_features(self.x)
        # self.y = torch.tensor([spk2class[spk] for spk in self.speakers]).to(self.device)
        # self.class2spk = dict(zip(spk2class.values(), spk2class.keys()))

    def __len__(self):
        return len(self.speakers)

    def __getitem__(self, index):
        embedding = self.normalize_embedding(self.x[index])
        # speaker_id = self.y[index]
        return embedding, torch.zeros([0])

    def normalize_embedding(self, vector):
        return torch.sub(vector, self.mean) / self.std

    def get_speaker(self, label):
        return self.class2spk[label]

    def get_embedding_dim(self):
        return self.x.shape[-1]

    def get_num_speaker(self):
        return len(torch.unique((self.y)))

    def set_labels(self, labels):
        self.y_old = self.y
        self.y = torch.full(size=(len(self),), fill_value=labels).to(self.device)
        # if isinstance(labels, int) or isinstance(labels, float):
        #    self.y = torch.full(size=len(self), fill_value=labels)
        # elif len(labels) == len(self):
        #    self.y = torch.tensor(labels)

    def _load_features(self, feature_path):
        if os.path.isfile(feature_path):
            vectors = torch.load(feature_path, map_location=self.device)
            if isinstance(vectors, list):
                vectors = torch.stack(vectors)

            self.mean = torch.mean(vectors)
            self.std = torch.std(vectors)
            return vectors, torch.zeros(vectors.size(0))
        else:
            vectors = torch.load(feature_path, map_location=self.device)

        self.mean = torch.mean(vectors)
        self.std = torch.std(vectors)

        spk2idx = {}
        with open(feature_path / f'{self.mode}2idx', 'r') as f:
            for line in f:
                split_line = line.strip().split()
                if len(split_line) == 2:
                    spk2idx[split_line[0].strip()] = int(split_line[1])

        speakers, indices = zip(*spk2idx.items())

        if (feature_path / 'utt2spk').exists():  # spk2idx contains utt_ids not speaker_ids
            utt2spk = {}
            with open(feature_path / 'utt2spk', 'r') as f:
                for line in f:
                    split_line = line.strip().split()
                    if len(split_line) == 2:
                        utt2spk[split_line[0].strip()] = split_line[1].strip()

            speakers = [utt2spk[utt] for utt in speakers]

        return vectors[np.array(indices)], speakers

    def _reformat_features(self, features):
        if len(features.shape) == 2:
            return features.reshape(features.shape[0], 1, 1, features.shape[1])
