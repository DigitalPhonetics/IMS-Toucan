import json

import torch

from ArticulatoryCombinedTextFrontend import ArticulatoryCombinedTextFrontend


def pretrain_512_dim():
    with open("embedding_table_512dim.json", 'r', encoding="utf8") as fp:
        datapoints = json.load(fp)
    datapoints.pop('̩')

    embedding = torch.nn.Sequential(torch.nn.Linear(66, 100),
                                    torch.nn.Tanh(),
                                    torch.nn.Linear(100, 512))
    loss_eucl_dist = torch.nn.L1Loss()
    loss_cos_sim = torch.nn.CosineSimilarity()
    optimizer = torch.optim.Adam(embedding.parameters())
    distances = list()

    tfe = ArticulatoryCombinedTextFrontend(language='en')

    for _ in range(4000):
        for phone in datapoints.keys():
            art_vec = torch.Tensor(tfe.phone_to_vector[phone]).unsqueeze(0)
            embedded_art_vec = embedding(art_vec)

            lookup_vec = torch.Tensor(datapoints[phone])

            distances.append(loss_eucl_dist(embedded_art_vec, lookup_vec) - loss_cos_sim(embedded_art_vec, lookup_vec))

            if len(distances) == 32:
                distance = sum(distances) / len(distances)
                distances = list()
                print(1 + distance)
                optimizer.zero_grad()
                distance.backward()
                optimizer.step()

    torch.save({"embedding_weights": embedding.state_dict()}, "embedding_pretrained_weights_combined_512dim.pt")


def pretrain_384_dim():
    with open("embedding_table_384dim.json", 'r', encoding="utf8") as fp:
        datapoints = json.load(fp)
    datapoints.pop('̩')

    embedding = torch.nn.Sequential(torch.nn.Linear(66, 100),
                                    torch.nn.Tanh(),
                                    torch.nn.Linear(100, 384))
    loss_eucl_dist = torch.nn.L1Loss()
    loss_cos_sim = torch.nn.CosineSimilarity()
    optimizer = torch.optim.Adam(embedding.parameters())
    distances = list()

    tfe = ArticulatoryCombinedTextFrontend(language='en')

    for _ in range(3000):
        for phone in datapoints.keys():
            art_vec = torch.Tensor(tfe.phone_to_vector[phone]).unsqueeze(0)
            embedded_art_vec = embedding(art_vec)

            lookup_vec = torch.Tensor(datapoints[phone])

            distances.append(loss_eucl_dist(embedded_art_vec, lookup_vec) - loss_cos_sim(embedded_art_vec, lookup_vec))

            if len(distances) == 32:
                distance = sum(distances) / len(distances)
                distances = list()
                print(1 + distance)
                optimizer.zero_grad()
                distance.backward()
                optimizer.step()

    torch.save({"embedding_weights": embedding.state_dict()}, "embedding_pretrained_weights_combined_384dim.pt")


if __name__ == '__main__':
    pretrain_384_dim()
