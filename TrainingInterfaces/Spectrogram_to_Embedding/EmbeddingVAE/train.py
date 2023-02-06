import os

import numpy
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from Model import Model

DEVICE = "cuda:7"


def train(epochs=130,
          net=Model(device=DEVICE),
          batch_size=64):
    torch.backends.cudnn.benchmark = True
    speaker_embedding_dataset = SpeakerEmbeddingDataset()
    dataloader = DataLoader(dataset=speaker_embedding_dataset,
                            batch_size=batch_size,
                            num_workers=10,
                            shuffle=True,
                            prefetch_factor=10,
                            persistent_workers=True,
                            drop_last=True)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
    create_eval_visualization(net, speaker_embedding_dataset, 0)
    for epoch in range(epochs):
        kl_losses = list()
        reconstruction_losses = list()
        for batch in tqdm(dataloader):
            _, kl_loss, reconstruction_loss = net(batch.to(DEVICE))
            if not torch.isnan(kl_loss):
                cyclic_kl_scale = ((epoch % 5) + 1) / 5000
                loss = kl_loss * cyclic_kl_scale + reconstruction_loss
                kl_losses.append(kl_loss.cpu().item())
            else:
                loss = reconstruction_loss
            kl_losses.append(kl_loss.cpu().item())
            reconstruction_losses.append(reconstruction_loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"KL this epoch: {sum(kl_losses) / len(kl_losses)}")
        print(f"Reconstruction this epoch: {sum(reconstruction_losses) / len(reconstruction_losses)}")
        create_eval_visualization(net, speaker_embedding_dataset, epoch + 1)
    torch.save({"model": net.state_dict()}, f="checkpoint.pt")


def create_eval_visualization(net, dataset, epoch):
    os.makedirs("validation", exist_ok=True)

    fig, axes = plt.subplots(1, 1, figsize=(6, 6))

    real_samples_shown = 2000
    fake_samples_shown = 2000

    generated_embeds = list()
    for _ in range(fake_samples_shown):
        generated_embeds.append(net().squeeze().detach().cpu().numpy())

    real_embeds = list()
    for _ in range(real_samples_shown):
        index = numpy.random.randint(0, len(dataset))
        real_embeds.append(dataset[index].numpy())

    reduction = TSNE(n_components=2,
                     learning_rate="auto",
                     init="pca",
                     n_jobs=-1)
    scaler = StandardScaler()

    embeddings_as_array = numpy.array(generated_embeds + real_embeds)
    dimensionality_reduced_embeddings = scaler.fit_transform(
        reduction.fit_transform(X=scaler.fit_transform(embeddings_as_array)))
    for i, datapoint in enumerate(dimensionality_reduced_embeddings):
        if i == 0:
            axes.scatter(x=datapoint[0],
                         y=datapoint[1],
                         c="b" if i < fake_samples_shown else "g",
                         label="fake",
                         alpha=0.4)
        elif i == fake_samples_shown:
            axes.scatter(x=datapoint[0],
                         y=datapoint[1],
                         c="b" if i < fake_samples_shown else "g",
                         label="real",
                         alpha=0.4)
        else:
            axes.scatter(x=datapoint[0],
                         y=datapoint[1],
                         c="b" if i < fake_samples_shown else "g",
                         alpha=0.4)
    axes.axis('off')
    axes.legend()
    plt.savefig(f"validation/{epoch}.png", bbox_inches='tight')
    plt.close()


class SpeakerEmbeddingDataset(Dataset):
    def __init__(self):
        self.embedding_list = torch.load("reference_embeddings_as_list.pt", map_location="cpu")
        print("loaded dataset")

    def __getitem__(self, index):
        return self.embedding_list[index]

    def __len__(self):
        return len(self.embedding_list)


if __name__ == '__main__':
    train()
