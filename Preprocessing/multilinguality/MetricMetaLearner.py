import json
import os
import pickle
import random

import kan
import torch
from tqdm import tqdm

from Modules.ToucanTTS.InferenceToucanTTS import ToucanTTS
from Utility.utils import load_json_from_path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from huggingface_hub import hf_hub_download

class MetricsCombiner(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.scoring_function = kan.KAN(width=[3, 5, 1], grid=5, k=5, seed=m, auto_save=False, device=DEVICE)
        self.scoring_function.speed(compile=True)

    def forward(self, x):
        return self.scoring_function(x)


class EnsembleModel(torch.nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = models

    def forward(self, x):
        x = x.to(DEVICE)
        distances = list()
        for model in self.models:
            distances.append(model(x))
        return sum(distances) / len(distances)


def create_learned_cache(model_path, cache_root="."):
    checkpoint = torch.load(model_path, map_location='cpu')
    embedding_provider = ToucanTTS(weights=checkpoint["model"], config=checkpoint["config"]).encoder.language_embedding
    embedding_provider.requires_grad_(False)
    language_list = load_json_from_path(hf_hub_download(repo_id="Flux9665/ToucanTTS", filename="supervised_languages.json"))
    tree_lookup_path = hf_hub_download(repo_id="Flux9665/ToucanTTS", filename="lang_1_to_lang_2_to_tree_dist.json")
    map_lookup_path = hf_hub_download(repo_id="Flux9665/ToucanTTS", filename="lang_1_to_lang_2_to_map_dist.json")
    asp_dict_path = hf_hub_download(repo_id="Flux9665/ToucanTTS", filename="asp_dict.pkl")
    if not os.path.exists(tree_lookup_path) or not os.path.exists(map_lookup_path):
        raise FileNotFoundError("Please ensure the caches exist!")
    if not os.path.exists(asp_dict_path):
        raise FileNotFoundError(f"{asp_dict_path} must be downloaded separately.")
    tree_dist = load_json_from_path(tree_lookup_path)
    map_dist = load_json_from_path(map_lookup_path)
    with open(asp_dict_path, 'rb') as dictfile:
        asp_sim = pickle.load(dictfile)
    lang_list = list(asp_sim.keys())
    largest_value_map_dist = 0.0
    for _, values in map_dist.items():
        for _, value in values.items():
            largest_value_map_dist = max(largest_value_map_dist, value)
    iso_codes_to_ids = load_json_from_path(hf_hub_download(repo_id="Flux9665/ToucanTTS", filename="iso_lookup.json"))[-1]
    train_set = language_list
    batch_size = 256
    model_list = list()
    print_intermediate_results = False

    # ensemble preparation
    n_models = 5
    print(f"Training ensemble of {n_models} models for learned distance metric.")
    for m in range(n_models):
        model_list.append(MetricsCombiner(m))
        optim = torch.optim.Adam(model_list[-1].parameters(), lr=0.0005)
        running_loss = list()
        for epoch in tqdm(range(25), desc=f"MetricsCombiner {m + 1}/{n_models} - Epoch"):
            for i in range(2000):
                # we have no dataloader, so first we build a batch
                embedding_distance_batch = list()
                metric_distance_batch = list()
                for _ in range(batch_size):
                    lang_1 = random.sample(train_set, 1)[0]
                    lang_2 = random.sample(train_set, 1)[0]
                    embedding_distance_batch.append(torch.nn.functional.mse_loss(embedding_provider(torch.LongTensor([iso_codes_to_ids[lang_1]])).squeeze(), embedding_provider(torch.LongTensor([iso_codes_to_ids[lang_2]])).squeeze()))
                    try:
                        _tree_dist = tree_dist[lang_2][lang_1]
                    except KeyError:
                        _tree_dist = tree_dist[lang_1][lang_2]
                    try:
                        _map_dist = map_dist[lang_2][lang_1] / largest_value_map_dist
                    except KeyError:
                        _map_dist = map_dist[lang_1][lang_2] / largest_value_map_dist
                    _asp_dist = 1.0 - asp_sim[lang_1][lang_list.index(lang_2)]
                    metric_distance_batch.append(torch.tensor([_tree_dist, _map_dist, _asp_dist], dtype=torch.float32))

                # ok now we have a batch prepared. Time to feed it to the model.
                scores = model_list[-1](torch.stack(metric_distance_batch).squeeze().to(DEVICE))
                if print_intermediate_results:
                    print("==================================")
                    print(scores.detach().squeeze()[:9])
                    print(torch.stack(embedding_distance_batch).squeeze()[:9])
                target_batch = torch.stack(embedding_distance_batch).squeeze().to(DEVICE)
                loss = torch.nn.functional.mse_loss(scores.squeeze(), target_batch, reduction="none")
                loss = loss / (target_batch + 0.0001)  # the smaller the distance, the higher the impact. We don't care about the large distances, only the nearest neighbors are important
                loss = loss.mean()

                optim.zero_grad()
                loss.backward()
                optim.step()
                running_loss.append(loss.cpu().item())

            print("\n\n")
            print(sum(running_loss) / len(running_loss))
            print("\n\n")
            running_loss = list()

        # import matplotlib.pyplot as plt
        # model_list[-1].scoring_function.plot(folder=f"kan_vis_{m}", beta=5000, in_vars=["Tree Distance", "Map Distance", "ASP Distance"], out_vars=["Predicted Distance between Embeddings"], scale=1.0, varscale=0.4)
        # plt.show()

    # Time to see if the final ensemble is any good
    ensemble = EnsembleModel(model_list)

    running_loss = list()
    for i in range(100):
        # we have no dataloader, so first we build a batch
        embedding_distance_batch = list()
        metric_distance_batch = list()
        for _ in range(batch_size):
            lang_1 = random.sample(train_set, 1)[0]
            lang_2 = random.sample(train_set, 1)[0]
            embedding_distance_batch.append(torch.nn.functional.mse_loss(embedding_provider(torch.LongTensor([iso_codes_to_ids[lang_1]])).squeeze(), embedding_provider(torch.LongTensor([iso_codes_to_ids[lang_2]])).squeeze()))
            try:
                _tree_dist = tree_dist[lang_2][lang_1]
            except KeyError:
                _tree_dist = tree_dist[lang_1][lang_2]
            try:
                _map_dist = map_dist[lang_2][lang_1] / largest_value_map_dist
            except KeyError:
                _map_dist = map_dist[lang_1][lang_2] / largest_value_map_dist
            _asp_dist = 1.0 - asp_sim[lang_1][lang_list.index(lang_2)]
            metric_distance_batch.append(torch.tensor([_tree_dist, _map_dist, _asp_dist], dtype=torch.float32))

        scores = ensemble(torch.stack(metric_distance_batch).squeeze())
        if print_intermediate_results:
            print("==================================")
            print(scores.detach().squeeze()[:9])
            print(torch.stack(embedding_distance_batch).squeeze()[:9])
        loss = torch.nn.functional.mse_loss(scores.squeeze(), torch.stack(embedding_distance_batch).squeeze().to(DEVICE))
        running_loss.append(loss.item())

    print("\n\n")
    print(sum(running_loss) / len(running_loss))

    language_to_language_to_learned_distance = dict()

    batch_accumulator = list()
    pair_accumulator = list()
    for lang_1 in tqdm(tree_dist):
        for lang_2 in tree_dist:
            try:
                if lang_2 in language_to_language_to_learned_distance:
                    if lang_1 in language_to_language_to_learned_distance[lang_2]:
                        continue  # it's symmetric
                if lang_1 not in language_to_language_to_learned_distance:
                    language_to_language_to_learned_distance[lang_1] = dict()
                try:
                    _tree_dist = tree_dist[lang_2][lang_1]
                except KeyError:
                    _tree_dist = tree_dist[lang_1][lang_2]
                try:
                    _map_dist = map_dist[lang_2][lang_1] / largest_value_map_dist
                except KeyError:
                    _map_dist = map_dist[lang_1][lang_2] / largest_value_map_dist
                _asp_dist = 1.0 - asp_sim[lang_1][lang_list.index(lang_2)]
                metric_distance = torch.tensor([_tree_dist, _map_dist, _asp_dist], dtype=torch.float32)
                batch_accumulator.append(metric_distance)
                pair_accumulator.append((lang_1, lang_2))
            except ValueError:
                continue
            except KeyError:
                continue
        if len(batch_accumulator) != 0:
            with torch.inference_mode():
                predicted_distances = ensemble(torch.stack(batch_accumulator))
            for predicted_distance, pair in zip(predicted_distances, pair_accumulator):
                language_to_language_to_learned_distance[pair[0]][pair[1]] = predicted_distance.item()
            batch_accumulator = list()
            pair_accumulator = list()

    with open(os.path.join(cache_root, 'lang_1_to_lang_2_to_learned_dist.json'), 'w', encoding='utf-8') as f:
        json.dump(language_to_language_to_learned_distance, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    create_learned_cache(hf_hub_download(repo_id="Flux9665/ToucanTTS", filename="ToucanTTS.pt"))
