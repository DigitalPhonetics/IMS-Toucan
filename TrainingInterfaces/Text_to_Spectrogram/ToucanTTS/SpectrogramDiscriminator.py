import torch
import torch.nn as nn


def weights_init_D(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class SpectrogramDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.D = DiscriminatorNet()
        self.D.apply(weights_init_D)

    def _generator_feature_matching(self, data_generated, data_real):
        for p in self.D.parameters():
            p.requires_grad = False  # freeze critic

        fmap_fake = self.D(data_generated)[1]
        fmap_real = self.D(data_real)[1]

        feature_matching_loss = 0.0
        for feat_fake, feat_real in zip(fmap_fake, fmap_real):
            feature_matching_loss += nn.functional.l1_loss(feat_fake, feat_real.detach())

        return feature_matching_loss

    def _discriminator_feature_matching(self, data_generated, data_real):
        for p in self.D.parameters():
            p.requires_grad = True  # unfreeze critic
        self.D.train()

        score_fake, fmap_fake = self.D(data_generated)
        score_real, fmap_real = self.D(data_real)

        feature_matching_loss = 0.0
        for feat_fake, feat_real in zip(fmap_fake, fmap_real):
            feature_matching_loss += nn.functional.l1_loss(feat_fake, feat_real.detach())

        feature_matching_loss += nn.functional.l1_loss(score_fake.mean(), torch.tensor(-1.0, device=score_fake.device))
        feature_matching_loss += nn.functional.l1_loss(score_real.mean(), torch.tensor(1.0, device=score_real.device))

        return feature_matching_loss * -1

    def calc_discriminator_loss(self, data_generated, data_real):
        return self._discriminator_feature_matching(data_generated.detach(), data_real)

    def calc_generator_feedback(self, data_generated, data_real):
        return self._generator_feature_matching(data_generated, data_real)


class DiscriminatorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.filters = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv2d(1, 32, kernel_size=(3, 9), padding=(1, 4))),
            nn.utils.weight_norm(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
            nn.utils.weight_norm(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
            nn.utils.weight_norm(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
            nn.utils.weight_norm(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ])

        self.out = nn.utils.weight_norm(nn.Conv2d(32, 1, 3, 1, 1))

        self.fc = nn.Linear(2000, 1)  # this needs to be changes everytime the window length is changes. It would be nice if this could be done dynamically.

    def forward(self, y):
        feature_maps = list()
        for d in self.filters:
            y = d(y)
            feature_maps.append(y)
            y = nn.functional.leaky_relu(y, 0.1)
        y = self.out(y)
        feature_maps.append(y)
        y = torch.flatten(y, 1, -1)
        y = self.fc(y)

        return y, feature_maps


if __name__ == '__main__':
    d = SpectrogramDiscriminator()
    fake = torch.randn([2, 200, 80])  # [Batch, Sequence Length, Spectrogram Buckets]
    real = torch.randn([2, 200, 80])  # [Batch, Sequence Length, Spectrogram Buckets]

    critic_loss = d.calc_discriminator_loss((fake.unsqueeze(1)), real.unsqueeze(1))
    generator_loss = d.calc_generator_feedback(fake.unsqueeze(1), real.unsqueeze(1))
    print(critic_loss)
    print(generator_loss)
