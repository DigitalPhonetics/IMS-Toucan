import os
import random

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from Preprocessing.articulatory_features import generate_feature_table
from Preprocessing.articulatory_features import get_feature_to_index_lookup
from Preprocessing.articulatory_features import get_phone_to_id
from TrainingInterfaces.Text_to_Spectrogram.AutoAligner.AlignerDataset import AlignerDataset


class ChunkedAlignerDataset(Dataset):

    def __init__(self,
                 path_to_transcript_dict,
                 cache_dir,
                 lang,
                 loading_processes,
                 cut_silences,
                 do_loudnorm,
                 device,
                 min_len_in_seconds=1,
                 max_len_in_seconds=15,
                 rebuild_cache=False,
                 verbose=False,
                 phone_input=False,
                 allow_unknown_symbols=False):
        self.num_chunks = (len(path_to_transcript_dict) // 10000) + 1
        self.chunk_list = list(range(self.num_chunks))
        if not os.path.exists(os.path.join(cache_dir, "aligner_train_cache.pt")):
            AlignerDataset(path_to_transcript_dict,
                           cache_dir=cache_dir,
                           lang=lang,
                           loading_processes=loading_processes,
                           min_len_in_seconds=min_len_in_seconds,
                           max_len_in_seconds=max_len_in_seconds,
                           verbose=verbose,
                           allow_unknown_symbols=allow_unknown_symbols,
                           cut_silences=cut_silences,
                           do_loudnorm=do_loudnorm,
                           device=device,
                           phone_input=phone_input)
        self.datapoint_feature_dump_list = torch.load(os.path.join(cache_dir, "aligner_train_cache.pt"), map_location='cpu')

        if not os.path.exists(os.path.join(cache_dir, "chunked_aligner_train_cache.pt")) or rebuild_cache:
            fisher_yates_shuffle(self.datapoint_feature_dump_list)
            elements_per_chunk = len(self.datapoint_feature_dump_list) // self.num_chunks
            for chunk_id in tqdm(self.chunk_list, desc="chunks", position=0):
                actual_chunk = list()
                for in_chunk_id in tqdm(range(elements_per_chunk), desc="elements in chunk", position=1, leave=False):
                    path_to_datapoint_file = self.datapoint_feature_dump_list[in_chunk_id + (chunk_id * elements_per_chunk)]
                    datapoint, speaker_embedding, filepath = torch.load(path_to_datapoint_file, map_location='cpu')
                    actual_chunk.append((datapoint, speaker_embedding, filepath))
                torch.save(actual_chunk, os.path.join(cache_dir, f"aligner_train_cache_chunk_{chunk_id}.pt"))
            print("deleting individual caches, since we now have chunked caches...")
            torch.save("successfully created", os.path.join(cache_dir, "chunked_aligner_train_cache.pt"))
            for path_for_mini_cache in tqdm(self.datapoint_feature_dump_list):
                os.remove(path_for_mini_cache)

        self.phone_to_vector = generate_feature_table()
        self.phone_to_id = get_phone_to_id()
        fisher_yates_shuffle(self.chunk_list)
        self.cache_dir = cache_dir
        self.active_chunk = self.chunk_list.pop()
        self.currently_loaded_datapoints = torch.load(os.path.join(cache_dir, f"aligner_train_cache_chunk_{self.active_chunk}.pt"), map_location='cpu')
        fisher_yates_shuffle(self.currently_loaded_datapoints)
        print(f"Prepared an Aligner dataset in {cache_dir}.")

    def text_vectors_to_id_sequence(self, text_vector):
        """
        duplicate code from the TextFrontend to avoid pickling errors
        """
        tokens = list()
        for vector in text_vector:
            if vector[get_feature_to_index_lookup()["word-boundary"]] == 0:
                # we don't include word boundaries when performing alignment, since they are not always present in audio.
                features = vector.cpu().numpy().tolist()
                if vector[get_feature_to_index_lookup()["vowel"]] == 1 and vector[get_feature_to_index_lookup()["nasal"]] == 1:
                    # for the sake of alignment, we ignore the difference between nasalized vowels and regular vowels
                    features[get_feature_to_index_lookup()["nasal"]] = 0
                features = features[13:]
                # the first 12 dimensions are for modifiers, so we ignore those when trying to find the phoneme in the ID lookup
                for phone in self.phone_to_vector:
                    if features == self.phone_to_vector[phone][13:]:
                        tokens.append(self.phone_to_id[phone])
                        # this is terribly inefficient, but it's fine
                        break
        return tokens

    def __getitem__(self, _):

        try:
            datapoint, speaker_embedding, filepath = self.currently_loaded_datapoints.pop()
        except IndexError:
            try:
                self.active_chunk = self.chunk_list.pop()
                self.currently_loaded_datapoints = torch.load(os.path.join(self.cache_dir, f"aligner_train_cache_chunk_{self.active_chunk}.pt"), map_location='cpu')
                fisher_yates_shuffle(self.currently_loaded_datapoints)
            except IndexError:
                self.chunk_list = list(range(self.num_chunks))
                self.active_chunk = self.chunk_list.pop()
                self.currently_loaded_datapoints = torch.load(os.path.join(self.cache_dir, f"aligner_train_cache_chunk_{self.active_chunk}.pt"), map_location='cpu')
                fisher_yates_shuffle(self.currently_loaded_datapoints)
            datapoint, speaker_embedding, filepath = self.currently_loaded_datapoints.pop()

        text_vector = datapoint[0]
        tokens = self.text_vectors_to_id_sequence(text_vector=text_vector)
        tokens = torch.LongTensor(tokens)
        return tokens, \
            torch.LongTensor([len(tokens)]), \
            datapoint[2], \
            datapoint[3], \
            speaker_embedding

    def __len__(self):
        return len(self.datapoint_feature_dump_list)


def fisher_yates_shuffle(lst):
    for i in range(len(lst) - 1, 0, -1):
        j = random.randint(0, i)
        lst[i], lst[j] = lst[j], lst[i]
