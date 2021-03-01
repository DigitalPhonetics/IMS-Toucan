import os
from multiprocessing import Process, Manager

import soundfile as sf
import torch
from torch.utils.data import Dataset

from PreprocessingForTTS.ProcessAudio import AudioPreprocessor
from PreprocessingForTTS.ProcessText import TextFrontend


class TransformerTTSDataset(Dataset):

    def __init__(self, path_to_transcript_dict, device=torch.device("cpu"), spemb=False, train=True,
                 loading_processes=40):
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
            process_list.append(Process(target=self.cache_builder_process, args=(key_split,)))
            process_list[-1].start()
        for process in process_list:
            process.join()

    def cache_builder_process(self, path_list):
        tf = TextFrontend(language="de",
                          use_panphon_vectors=False,
                          use_shallow_pos=False,
                          use_sentence_type=False,
                          use_positional_information=False,
                          use_word_boundaries=False,
                          use_chinksandchunks_ipb=False,
                          use_explicit_eos=True)
        _, sr = sf.read(os.path.join("Corpora/CSS10/", path_list[0]))
        ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024)
        for path in path_list:
            transcript = self.path_to_transcript_dict[path]
            wave, _ = sf.read(os.path.join("Corpora/CSS10/", path))
            if 50000 < len(wave) < 230000:
                print("processing {}".format(path))
                cached_text = tf.string_to_tensor(transcript).long()
                cached_text_lens = torch.LongTensor([len(cached_text)])
                cached_speech = ap.audio_to_mel_spec_tensor(wave).transpose(0, 1)
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
