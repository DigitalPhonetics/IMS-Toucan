import json
import os
from multiprocessing import Process, Manager

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

from PreprocessingForTTS.ProcessAudio import AudioPreprocessor
from PreprocessingForTTS.ProcessText import TextFrontend


class TransformerTTSDataset(Dataset):

    def __init__(self, path_to_transcript_dict,
                 spemb=False,
                 train=True,
                 loading_processes=4,
                 cache_dir=os.path.join("Corpora", "CSS10_DE"),
                 lang="de",
                 min_len_in_seconds=1,
                 max_len_in_seconds=20,
                 cut_silences=False,
                 rebuild_cache=False):
        self.spemb = spemb
        if ((not os.path.exists(os.path.join(cache_dir, "trans_train_cache.json"))) and train) or (
                (not os.path.exists(os.path.join(cache_dir, "trans_valid_cache.json"))) and (not train)) or \
                rebuild_cache:
            ressource_manager = Manager()
            self.path_to_transcript_dict = ressource_manager.dict(path_to_transcript_dict)
            all_keys_ordered = list(self.path_to_transcript_dict.keys())
            if train:
                key_list = all_keys_ordered[:-100]
            else:
                key_list = all_keys_ordered[-100:]

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
                    Process(target=self.cache_builder_process,
                            args=(key_split, spemb, lang, min_len_in_seconds, max_len_in_seconds, cut_silences),
                            daemon=True))
                process_list[-1].start()
            for process in process_list:
                process.join()
            self.datapoints = list(self.datapoints)
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
        print("Prepared {} datapoints.".format(len(self.datapoints)))

    def cache_builder_process(self, path_list, spemb, lang, min_len, max_len, cut_silences):
        tf = TextFrontend(language=lang,
                          use_panphon_vectors=False,
                          use_word_boundaries=False,
                          use_explicit_eos=False,
                          use_prosody=False)
        _, sr = sf.read(path_list[0])
        if spemb:
            wav2mel = torch.jit.load("Models/Use/SpeakerEmbedding/wav2mel.pt")
            dvector = torch.jit.load("Models/Use/SpeakerEmbedding/dvector-step250000.pt").eval()
        ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024,
                               cut_silence=cut_silences)
        for index, path in enumerate(path_list):
            transcript = self.path_to_transcript_dict[path]
            wave, sr = sf.read(path)
            if min_len <= len(wave) / sr <= max_len:
                print("Processing {} out of {}.".format(index, len(path_list)))
                cached_text = tf.string_to_tensor(transcript).squeeze(0).numpy().tolist()
                cached_text_lens = len(cached_text)
                cached_speech = ap.audio_to_mel_spec_tensor(wave).transpose(0, 1).numpy().tolist()
                cached_speech_lens = len(cached_speech)
                if spemb:
                    wav_tensor, sample_rate = torchaudio.load(path)
                    mel_tensor = wav2mel(wav_tensor, sample_rate)
                    emb_tensor = dvector.embed_utterance(mel_tensor)
                    cached_spemb = emb_tensor.detach().numpy().tolist()
                    self.datapoints.append(
                        [cached_text, cached_text_lens, cached_speech, cached_speech_lens, cached_spemb])
                else:
                    self.datapoints.append([cached_text, cached_text_lens, cached_speech, cached_speech_lens])

    def __getitem__(self, index):
        if not self.spemb:
            return self.datapoints[index][0], \
                   self.datapoints[index][1], \
                   self.datapoints[index][2], \
                   self.datapoints[index][3]
        else:
            return self.datapoints[index][0], \
                   self.datapoints[index][1], \
                   self.datapoints[index][2], \
                   self.datapoints[index][3], \
                   self.datapoints[index][4]

    def __len__(self):
        return len(self.datapoints)
