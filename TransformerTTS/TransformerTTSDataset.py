import os
from threading import Thread

import soundfile as sf
import torch
from torch.utils.data import Dataset

from PreprocessingForTTS.ProcessAudio import AudioPreprocessor
from PreprocessingForTTS.ProcessText import TextFrontend


class TransformerTTSDataset(Dataset):

    def __init__(self, path_to_transcript_dict, device=torch.device("cpu"), spemb=False, train=True,
                 loading_threads=16):
        self.path_to_transcript_dict = path_to_transcript_dict
        if train:
            key_list = list(self.path_to_transcript_dict.keys())[:-100]
        else:
            key_list = list(self.path_to_transcript_dict.keys())[-100:]
        self.spemb = spemb
        self.device = device
        self.tf = TextFrontend(language="de",
                               use_panphon_vectors=False,
                               use_shallow_pos=False,
                               use_sentence_type=False,
                               use_positional_information=False,
                               use_word_boundaries=False,
                               use_chinksandchunks_ipb=False,
                               use_explicit_eos=True)
        _, sr = sf.read(os.path.join("Corpora/CSS10/", key_list[0]))
        self.ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024)
        # build cache
        print("... building dataset cache ...")
        self.datapoints = list()
        # make threads
        key_splits = list()
        thread_list = list()
        for i in range(loading_threads):
            key_splits.append(key_list[i * len(key_list) / loading_threads:i + 1 * len(key_list) / loading_threads])
        for key_split in key_splits:
            thread_list.append(Thread(target=self.cache_builder_thread, args=(key_split,)))
            thread_list[-1].start()
        for thread in thread_list:
            thread.join()

    def cache_builder_thread(self, path_list):
        for path in path_list:
            transcript = self.path_to_transcript_dict[path]
            wave, _ = sf.read(os.path.join("Corpora/CSS10/", path))
            if 50000 < len(wave) < 230000:
                print("processing {}".format(path))
                cached_text = self.tf.string_to_tensor(transcript).long()
                cached_text_lens = torch.LongTensor([len(cached_text)])
                cached_speech = self.ap.audio_to_mel_spec_tensor(wave).transpose(0, 1)
                cached_speech_lens = torch.LongTensor([len(cached_speech)])
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
