import os.path
import pickle

import torch

from Preprocessing.multilinguality.SimilaritySolver import load_json_from_path
from Preprocessing.multilinguality.create_map_and_tree_dist_lookups import CacheCreator


class LESL(torch.nn.Module):

    def __init__(self):
        super().__init__()
        cc = CacheCreator(cache_root="Preprocessing/multilinguality")
        if not os.path.exists('Preprocessing/multilinguality/lang_1_to_lang_2_to_tree_dist.json'):
            cc.create_tree_cache(cache_root="Preprocessing/multilinguality")
        if not os.path.exists('Preprocessing/multilinguality/lang_1_to_lang_2_to_tree_dist.json'):
            cc.create_map_cache(cache_root="Preprocessing/multilinguality")
        if not os.path.exists("Preprocessing/multilinguality/asp_dict.pkl"):
            print("download asp file")  # TODO downloader script with release

        self.tree_dist = load_json_from_path('Preprocessing/multilinguality/lang_1_to_lang_2_to_tree_dist.json')
        self.map_dist = load_json_from_path('Preprocessing/multilinguality/lang_1_to_lang_2_to_map_dist.json')
        with open("Preprocessing/multilinguality/asp_dict.pkl", 'rb') as dictfile:
            self.asp_sim = pickle.load(dictfile)
        self.lang_list = list(self.asp_sim.keys())  # list of all languages, to get lang_b's index

        self.largest_value_tree_dist = 0.0
        for _, values in self.tree_dist.items():
            for _, value in values.items():
                self.largest_value_tree_dist = max(self.largest_value_tree_dist, value)
        self.largest_value_map_dist = 0.0
        for _, values in self.map_dist.items():
            for _, value in values.items():
                self.largest_value_map_dist = max(self.largest_value_map_dist, value)

        self.dist = torch.nn.L1Loss()

        iso_codes_to_ids = load_json_from_path("Preprocessing/multilinguality/iso_lookup.json")[-1]
        self.ids_to_iso_codes = {v: k for k, v in iso_codes_to_ids.items()}

    def forward(self, language_ids, language_embeddings):
        """
        Args:
            language_ids (Tensor): IDs of languages in the same order as the embeddings to calculate the distances according to the metrics.
            language_embeddings (Tensor): Batch of language embeddings, of which the distances will be compared to the distances according to the metrics.

        Returns:
            Tensor: Language Embedding Structure Loss Value
        """

        losses = list()
        for language_id_1, language_embedding_1 in zip(language_ids, language_embeddings):
            for language_id_2, language_embedding_2 in zip(language_ids, language_embeddings):
                if language_id_1 != language_id_2:
                    embed_dist = self.dist(language_embedding_1, language_embedding_2)
                    lang_1 = self.ids_to_iso_codes[language_id_1.cpu().item()]
                    lang_2 = self.ids_to_iso_codes[language_id_2.cpu().item()]

                    # Value Range Normalized Tree Dist
                    tree_dist = self.tree_dist[lang_1][lang_2] / self.largest_value_tree_dist

                    # Value Range Normalized Map Dist
                    map_dist = self.map_dist[lang_1][lang_2] / self.largest_value_map_dist

                    # Value Range Normalized ASP Dist
                    lang_2_idx = self.lang_list.index(lang_2)
                    asp_dist = 1.0 - self.asp_sim[lang_1][lang_2_idx]  # it's a similarity measure that goes from 0 to 1, so we subtract it from 1 to turn it into a distance

                    # Total distance should be similar to bring some structure into the embedding-space
                    metric_dist = (tree_dist + map_dist + asp_dist) / 3
                    losses.append(self.dist(embed_dist, metric_dist))

        return sum(losses) / len(losses)