"""
A relation network approach at speaker embedding
"""

import torch

from SpeakerEmbedding.ContrastiveLoss import ContrastiveLoss


class SiameseSpeakerEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, (20, 20))
        self.acti1 = torch.nn.LeakyReLU()
        self.drop1 = torch.nn.Dropout2d(0.2)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(1, 1, (20, 20))
        self.acti2 = torch.nn.LeakyReLU()
        self.drop2 = torch.nn.Dropout2d(0.2)
        self.pool2 = torch.nn.MaxPool2d(2)
        self.conv3 = torch.nn.Conv2d(1, 1, (20, 20))
        self.acti3 = torch.nn.LeakyReLU()
        self.drop3 = torch.nn.Dropout2d(0.2)
        self.encoder = torch.nn.Sequential(self.conv1, self.acti1, self.drop1, self.pool1,
                                           self.conv2, self.acti2, self.drop2, self.pool2,
                                           self.conv3, self.acti3, self.drop3)
        self.expander = torch.nn.Sequential(torch.nn.Linear(94, 256),
                                            torch.nn.Tanh())
        self.comparator = torch.nn.CosineSimilarity()
        self.criterion = ContrastiveLoss()

    def forward(self, sample1, sample2, label):
        """
        :param sample1: batch of spectrograms with 512 buckets
        :param sample2: batch of spectrograms with 512 buckets
        :param label: batch of distance labels (-1 means same class and +1 means different class)
        :return: loss to optimize for
        """
        self.encoder.train()
        encoded1 = self.encoder(sample1)
        encoded2 = self.encoder(sample2)
        # reduce over sequence axis
        vector1 = self.expander(torch.mean(encoded1, 3))
        vector2 = self.expander(torch.mean(encoded2, 3))
        # get similarity
        sim = self.comparator(vector1, vector2)
        dist = torch.neg(sim)
        loss = self.criterion(dist, label)
        return loss

    def inference(self, sample):
        """
        :param sample: spectrogram to be embedded (512 buckets)
        :return: embedding for speaker
        """
        self.encoder.eval()
        return self.expander(torch.mean(self.encoder(sample), 3))
