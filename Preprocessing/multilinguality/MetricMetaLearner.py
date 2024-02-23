import json
import os
import pickle
import random

import torch
from tqdm import tqdm

from Architectures.ToucanTTS.InferenceToucanTTS import ToucanTTS
from Preprocessing.multilinguality.SimilaritySolver import load_json_from_path
from Utility.storage_config import MODELS_DIR


class MetricsCombiner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scoring_function = torch.nn.Sequential(torch.nn.Linear(3, 8),
                                                    torch.nn.Tanh(),
                                                    torch.nn.Linear(8, 8),
                                                    torch.nn.Tanh(),
                                                    torch.nn.Linear(8, 1))

    def forward(self, x):
        return self.scoring_function(x)


class EnsembleModel(torch.nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = models

    def forward(self, x):
        distances = list()
        for model in self.models:
            distances.append(model(x))
        return sum(distances) / len(distances)


checkpoint = torch.load(os.path.join(MODELS_DIR, f"ToucanTTS_Meta", "best.pt"), map_location='cpu')  # this assumes that MODELS_DIR is an absolute path, the relative path will fail at this location
embedding_provider = ToucanTTS(weights=checkpoint["model"], config=checkpoint["config"]).encoder.language_embedding
embedding_provider.requires_grad_(False)
language_list = ['eng', 'deu', 'fra', 'spa', 'cmn', 'por', 'pol', 'ita', 'nld', 'ell', 'fin', 'vie', 'rus', 'hun', 'bem', 'swh', 'amh', 'wol', 'mal', 'chv', 'iba', 'jav', 'fon', 'hau', 'lbb', 'kik', 'lin', 'lug', 'luo', 'sxb', 'yor', 'nya', 'loz', 'toi', 'afr', 'arb', 'asm', 'ast', 'azj', 'bel', 'bul', 'ben', 'bos', 'cat', 'ceb', 'sdh',
                 'ces', 'cym', 'dan', 'ekk', 'pes', 'fil', 'gle', 'glg', 'guj', 'heb', 'hin', 'hrv', 'hye', 'ind', 'ibo', 'isl', 'kat', 'kam', 'kea', 'kaz', 'khm', 'kan', 'kor', 'ltz', 'lao', 'lit', 'lvs', 'mri', 'mkd', 'xng', 'mar', 'zsm', 'mlt', 'oci', 'ory', 'pan', 'pst', 'ron', 'snd', 'slk', 'slv', 'sna', 'som', 'srp', 'swe', 'tam',
                 'tel', 'tgk', 'tur', 'ukr', 'umb', 'urd', 'uzn', 'bhd', 'kfs', 'dgo', 'gbk', 'bgc', 'xnr', 'kfx', 'mjl', 'bfz', 'acf', 'bss', 'inb', 'nca', 'quh', 'wap', 'acr', 'bus', 'dgr', 'maz', 'nch', 'qul', 'tav', 'wmw', 'acu', 'byr', 'dik', 'iou', 'mbb', 'ncj', 'qvc', 'tbc', 'xed', 'agd', 'bzh', 'djk', 'ipi', 'mbc', 'ncl', 'qve',
                 'tbg', 'xon', 'agg', 'bzj', 'dop', 'jac', 'mbh', 'ncu', 'qvh', 'tbl', 'xtd', 'agn', 'caa', 'jic', 'mbj', 'ndj', 'qvm', 'tbz', 'xtm', 'agr', 'cab', 'emp', 'jiv', 'mbt', 'nfa', 'qvn', 'tca', 'yaa', 'agu', 'cap', 'jvn', 'mca', 'ngp', 'qvs', 'tcs', 'yad', 'aia', 'car', 'ese', 'mcb', 'ngu', 'qvw', 'yal', 'cax', 'kaq', 'mcd',
                 'nhe', 'qvz', 'tee', 'ycn', 'ake', 'cbc', 'far', 'mco', 'qwh', 'yka', 'alp', 'cbi', 'kdc', 'mcp', 'nhu', 'qxh', 'ame', 'cbr', 'gai', 'kde', 'mcq', 'nhw', 'qxn', 'tew', 'yre', 'amf', 'cbs', 'gam', 'kdl', 'mdy', 'nhy', 'qxo', 'tfr', 'yva', 'amk', 'cbt', 'geb', 'kek', 'med', 'nin', 'rai', 'zaa', 'apb', 'cbu', 'glk', 'ken',
                 'mee', 'nko', 'rgu', 'zab', 'apr', 'cbv', 'meq', 'tgo', 'zac', 'arl', 'cco', 'gng', 'kje', 'met', 'nlg', 'rop', 'tgp', 'zad', 'grc', 'klv', 'mgh', 'nnq', 'rro', 'zai', 'ata', 'cek', 'gub', 'kmu', 'mib', 'noa', 'ruf', 'tna', 'zam', 'atb', 'cgc', 'guh', 'kne', 'mie', 'not', 'rug', 'tnk', 'zao', 'atg', 'chf', 'knf', 'mih',
                 'npl', 'tnn', 'zar', 'awb', 'chz', 'gum', 'knj', 'mil', 'sab', 'tnp', 'zas', 'cjo', 'guo', 'ksr', 'mio', 'obo', 'seh', 'toc', 'zav', 'azg', 'cle', 'gux', 'kue', 'mit', 'omw', 'sey', 'tos', 'zaw', 'azz', 'cme', 'gvc', 'kvn', 'miz', 'ood', 'sgb', 'tpi', 'zca', 'bao', 'cni', 'gwi', 'kwd', 'mkl', 'shp', 'tpt', 'zga', 'bba',
                 'cnl', 'gym', 'kwf', 'mkn', 'ote', 'sja', 'trc', 'ziw', 'bbb', 'cnt', 'gyr', 'kwi', 'mop', 'otq', 'snn', 'ttc', 'zlm', 'cof', 'hat', 'kyc', 'mox', 'pab', 'snp', 'tte', 'zos', 'bgt', 'con', 'kyf', 'mpm', 'pad', 'tue', 'zpc', 'bjr', 'cot', 'kyg', 'mpp', 'soy', 'tuf', 'zpl', 'bjv', 'cpa', 'kyq', 'mpx', 'pao', 'tuo', 'zpm',
                 'bjz', 'cpb', 'hlt', 'kyz', 'mqb', 'pib', 'spp', 'zpo', 'bkd', 'cpu', 'hns', 'lac', 'mqj', 'pir', 'spy', 'txq', 'zpu', 'blz', 'crn', 'hto', 'lat', 'msy', 'pjt', 'sri', 'txu', 'zpz', 'bmr', 'cso', 'hub', 'lex', 'mto', 'pls', 'srm', 'udu', 'ztq', 'bmu', 'ctu', 'lgl', 'muy', 'poi', 'srn', 'zty', 'bnp', 'cuc', 'lid', 'mxb',
                 'stp', 'upv', 'zyp', 'boa', 'cui', 'huu', 'mxq', 'sus', 'ura', 'boj', 'cuk', 'huv', 'llg', 'mxt', 'poy', 'suz', 'urb', 'box', 'cwe', 'hvn', 'prf', 'urt', 'bpr', 'cya', 'ign', 'lww', 'myk', 'ptu', 'usp', 'bps', 'daa', 'ikk', 'maj', 'myy', 'vid', 'bqc', 'dah', 'nab', 'qub', 'tac', 'bqp', 'ded', 'imo', 'maq', 'nas', 'quf',
                 'taj', 'vmy']
tree_dist = load_json_from_path('lang_1_to_lang_2_to_tree_dist.json')
map_dist = load_json_from_path('lang_1_to_lang_2_to_map_dist.json')
with open("asp_dict.pkl", 'rb') as dictfile:
    asp_sim = pickle.load(dictfile)
lang_list = list(asp_sim.keys())
largest_value_map_dist = 0.0
for _, values in map_dist.items():
    for _, value in values.items():
        largest_value_map_dist = max(largest_value_map_dist, value)
iso_codes_to_ids = load_json_from_path("iso_lookup.json")[-1]
ids_to_iso_codes = {v: k for k, v in iso_codes_to_ids.items()}
train_set = language_list
batch_size = 128
model_list = list()
print_intermediate_results = False

# ensemble preparation
for _ in range(10):
    model_list.append(MetricsCombiner())
    model_list[-1].train()
    optim = torch.optim.Adam(model_list[-1].parameters(), lr=0.00005)
    running_loss = list()
    for epoch in tqdm(range(35)):
        for i in range(1000):
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
            scores = model_list[-1](torch.stack(metric_distance_batch).squeeze())
            if print_intermediate_results:
                print("==================================")
                print(scores.detach().squeeze()[:9])
                print(torch.stack(embedding_distance_batch).squeeze()[:9])
            loss = torch.nn.functional.mse_loss(scores.squeeze(), torch.stack(embedding_distance_batch).squeeze(), reduction="none")
            loss = loss / (torch.stack(embedding_distance_batch).squeeze() + 0.0001)
            loss = loss.mean()

            running_loss.append(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()

        print("\n\n")
        print(sum(running_loss) / len(running_loss))
        print("\n\n")
        running_loss = list()

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
    print("==================================")
    print(scores.detach().squeeze()[:9])
    print(torch.stack(embedding_distance_batch).squeeze()[:9])
    loss = torch.nn.functional.mse_loss(scores.squeeze(), torch.stack(embedding_distance_batch).squeeze())
    running_loss.append(loss.item())

print("\n\n")
print(sum(running_loss) / len(running_loss))

language_to_language_to_learned_distance = dict()

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
            with torch.inference_mode():
                predicted_distance = ensemble(metric_distance)
            language_to_language_to_learned_distance[lang_1][lang_2] = predicted_distance.item()
        except ValueError:
            continue
        except KeyError:
            continue

with open('lang_1_to_lang_2_to_learned_dist_information_leak_fixed_3.json', 'w', encoding='utf-8') as f:
    json.dump(language_to_language_to_learned_distance, f, ensure_ascii=False, indent=4)
