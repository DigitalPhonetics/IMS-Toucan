"""
A relation network approach at speaker embedding
"""

import os

import torch

from SpeakerEmbedding.ContrastiveLoss import ContrastiveLoss


class SiameseSpeakerEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(15, 15))
        self.act1 = torch.nn.LeakyReLU()
        self.drop1 = torch.nn.Dropout2d(0.1)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(10, 10))
        self.act2 = torch.nn.LeakyReLU()
        self.drop2 = torch.nn.Dropout2d(0.1)
        self.pool2 = torch.nn.MaxPool2d(2)
        self.conv3 = torch.nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(5, 5))
        self.act3 = torch.nn.LeakyReLU()
        self.drop3 = torch.nn.Dropout2d(0.1)

        self.encoder_fine = torch.nn.Sequential(self.conv1, self.act1, self.drop1)
        self.encoder_medium = torch.nn.Sequential(self.pool1, self.conv2, self.act2, self.drop2)
        self.encoder_coarse = torch.nn.Sequential(self.pool2, self.conv3, self.act3, self.drop3)

        self.channel_reducer_1 = torch.nn.Conv2d(in_channels=10, out_channels=1, kernel_size=1)
        self.channel_reducer_2 = torch.nn.Conv2d(in_channels=10, out_channels=1, kernel_size=1)
        self.channel_reducer_3 = torch.nn.Conv2d(in_channels=10, out_channels=1, kernel_size=1)

        self.expander = torch.nn.Sequential(torch.nn.Linear(53, 128), torch.nn.Tanh())

        self.similarity = torch.nn.CosineSimilarity()
        self.criterion = ContrastiveLoss()

    def forward(self, sample1, sample2, label):
        """
        :param sample1: batch of mel banks with 80 buckets
        :param sample2: batch of mel banks with 80 buckets
        :param label: batch of distance labels (-1 means same class and +1 means different class)
        :return: loss to optimize for
        """
        # encode both samples
        encoded_fine_1 = self.encoder_fine(sample1)
        encoded_fine_2 = self.encoder_fine(sample2)

        encoded_medium_1 = self.encoder_medium(encoded_fine_1)
        encoded_medium_2 = self.encoder_medium(encoded_fine_2)

        encoded_coarse_1 = self.encoder_coarse(encoded_medium_1)
        encoded_coarse_2 = self.encoder_coarse(encoded_medium_2)

        # reduce over sequence axis
        ef1_vec = torch.mean(self.channel_reducer_1(encoded_fine_1), 3)
        ef2_vec = torch.mean(self.channel_reducer_1(encoded_fine_2), 3)

        em1_vec = torch.mean(self.channel_reducer_2(encoded_medium_1), 3)
        em2_vec = torch.mean(self.channel_reducer_2(encoded_medium_2), 3)

        ec1_vec = torch.mean(self.channel_reducer_3(encoded_coarse_1), 3)
        ec2_vec = torch.mean(self.channel_reducer_3(encoded_coarse_2), 3)

        # expand dimensions
        info1 = torch.cat([ef1_vec, em1_vec, ec1_vec], 2)
        info2 = torch.cat([ef2_vec, em2_vec, ec2_vec], 2)

        embedding1 = self.expander(info1)
        embedding2 = self.expander(info2)

        # get distance
        cos_sim = self.similarity(embedding1, embedding2)
        cos_dist = torch.neg(cos_sim)

        # get loss
        loss = self.criterion(cos_dist, label)

        return loss

    def inference(self, sample):
        """
        :param sample: mel bank to be embedded (80 buckets)
        :return: embedding for speaker
        """
        # encode sample
        encoded_fine_1 = self.encoder_fine(sample)
        encoded_medium_1 = self.encoder_medium(encoded_fine_1)
        encoded_coarse_1 = self.encoder_coarse(encoded_medium_1)

        # reduce over sequence axis
        ef1_vec = torch.mean(self.channel_reducer_1(encoded_fine_1), 3)
        em1_vec = torch.mean(self.channel_reducer_2(encoded_medium_1), 3)
        ec1_vec = torch.mean(self.channel_reducer_3(encoded_coarse_1), 3)

        # expand dimensions
        info1 = torch.cat([ef1_vec, em1_vec, ec1_vec], 2)
        embedding = self.expander(info1)
        return embedding

    def get_conf(self):
        return "SiameseSpeakerEmbedding"


def build_spk_emb_model():
    model = SiameseSpeakerEmbedding()
    params = torch.load(os.path.join("Models", "Use", "SpeakerEmbedding.pt"))["model"]
    model.load_state_dict(params)
    model.eval()
    return model
