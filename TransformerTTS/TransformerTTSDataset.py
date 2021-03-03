import json
import os
from multiprocessing import Process, Manager

import soundfile as sf
import torch
from torch.utils.data import Dataset

from PreprocessingForTTS.ProcessAudio import AudioPreprocessor
from PreprocessingForTTS.ProcessText import TextFrontend


class TransformerTTSDataset(Dataset):

    def __init__(self, path_to_transcript_dict,
                 device=torch.device("cpu"),
                 spemb=False,
                 train=True,
                 loading_processes=4,
                 save=True,
                 load=False,
                 cache_dir=os.path.join("Corpora", "CSS10"),
                 lang="en",
                 min_len=50000,
                 max_len=230000):
        if not load:
            ressource_manager = Manager()
            self.path_to_transcript_dict = ressource_manager.dict(path_to_transcript_dict)
            if train:
                key_list = list(self.path_to_transcript_dict.keys())[:-100]
            else:
                key_list = list(self.path_to_transcript_dict.keys())[-100:]
            self.spemb = spemb
            self.device = device

            # build cache
            print("... building dataset cache ...")
            self.datapoints = ressource_manager.list()
            # make processes
            key_splits = list()
            process_list = list()
            for i in range(loading_processes):
                key_splits.append(
                    key_list[i * len(key_list) // loading_processes:(i + 1) * len(key_list) // loading_processes])
            for key_split in key_splits:
                process_list.append(
                    Process(target=self.cache_builder_process, args=(key_split, lang, min_len, max_len), daemon=True))
                process_list[-1].start()
            for process in process_list:
                process.join()
            self.datapoints = list(self.datapoints)
            if save:
                # save to json so we can rebuild cache quickly
                if train:
                    with open(os.path.join(cache_dir, "trans_train_cache.json"), 'w') as fp:
                        json.dump(self.datapoints, fp)
                else:
                    with open(os.path.join(cache_dir, "trans_valid_cache.json"), 'w') as fp:
                        json.dump(self.datapoints, fp)
        else:
            # just load the datapoints
            if train:
                with open(os.path.join(cache_dir, "trans_train_cache.json"), 'r') as fp:
                    self.datapoints = json.load(fp)
            else:
                with open(os.path.join(cache_dir, "trans_valid_cache.json"), 'r') as fp:
                    self.datapoints = json.load(fp)

    def cache_builder_process(self, path_list, lang, min_len, max_len):
        tf = TextFrontend(language=lang,
                          use_panphon_vectors=False,
                          use_shallow_pos=False,
                          use_sentence_type=False,
                          use_positional_information=False,
                          use_word_boundaries=False,
                          use_chinksandchunks_ipb=False,
                          use_explicit_eos=True)
        _, sr = sf.read(path_list[0])
        ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024)
        for path in path_list:
            transcript = self.path_to_transcript_dict[path]
            wave, _ = sf.read(path)
            if min_len < len(wave) < max_len:
                print("processing {}".format(path))
                cached_text = tf.string_to_tensor(transcript).numpy().tolist()
                cached_text_lens = len(cached_text)
                cached_speech = ap.audio_to_mel_spec_tensor(wave).transpose(0, 1).numpy().tolist()
                cached_speech_lens = len(cached_speech)
                self.datapoints.append([cached_text, cached_text_lens, cached_speech, cached_speech_lens])
                if self.spemb:
                    print("not implemented yet")
                    raise NotImplementedError

    def __getitem__(self, index):
        return self.datapoints[index][0], \
               self.datapoints[index][1], \
               self.datapoints[index][2], \
               self.datapoints[index][3]

    def __len__(self):
        return len(self.datapoints)
