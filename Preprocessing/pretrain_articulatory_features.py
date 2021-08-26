import json

import panphon
import torch

with open("embedding_table.json", 'r', encoding="utf8") as fp:
    datapoints = json.load(fp)
datapoints.pop("?")
datapoints.pop(".")
datapoints.pop("!")
datapoints.pop('̩')
datapoints.pop("ᵻ")
datapoints.pop("ɚ")

feature_table = panphon.FeatureTable()

embedding = torch.nn.Sequential(torch.nn.Linear(25, 50),
                                torch.nn.Tanh(),
                                torch.nn.Linear(50, 512))
loss_eucl_dist = torch.nn.L1Loss()
loss_cos_sim = torch.nn.CosineSimilarity()
optimizer = torch.optim.Adam(embedding.parameters())
distances = list()

for _ in range(6000):
    for phone in datapoints.keys():
        if phone == "~":
            art_vec = torch.Tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        else:
            art_vec = feature_table.word_to_vector_list(phone, numeric=True)
            art_vec[0].append(0)
            art_vec = torch.Tensor(art_vec)
        lookup_vec = torch.Tensor(datapoints[phone])

        embedded_art_vec = embedding(art_vec)
        distances.append(loss_eucl_dist(embedded_art_vec, lookup_vec) - loss_cos_sim(embedded_art_vec, lookup_vec))

        if len(distances) == 30:
            distance = sum(distances) / len(distances)
            distances = list()
            print(1 + distance)
            optimizer.zero_grad()
            distance.backward()
            optimizer.step()

torch.save({"embedding_weights": embedding.state_dict()}, "embedding_pretrained_weights.pt")
