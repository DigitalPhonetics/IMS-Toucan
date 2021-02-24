import os

import soundfile as sf
import torch
from torch.utils.data import Dataset

from PreprocessingForTTS.ProcessAudio import AudioPreprocessor
from PreprocessingForTTS.ProcessText import TextFrontend


class TransformerTTSDataset(Dataset):

    def __init__(self, path_to_transcript_dict, device=torch.device("cpu"), spemb=False, train=True):
        self.path_to_transcript_dict = path_to_transcript_dict
        if train:
            key_list = list(self.path_to_transcript_dict.keys())[:-100]
        else:
            key_list = list(self.path_to_transcript_dict.keys())[-100:]
        self.spemb = spemb
        self.device = device
        tf = TextFrontend(language="de",
                          use_panphon_vectors=False,
                          use_shallow_pos=False,
                          use_sentence_type=False,
                          use_positional_information=False,
                          use_word_boundaries=False,
                          use_chinksandchunks_ipb=False,
                          use_explicit_eos=True)
        ap = None
        # build cache
        print("... building dataset cache ...")
        self.cached_text = list()
        self.cached_text_lens = list()
        self.cached_speech = list()
        self.cached_speech_lens = list()
        for path in key_list:
            transcript = self.path_to_transcript_dict[path]
            wave, sr = sf.read(os.path.join("Corpora/CSS10/", path))
            if 50000 < len(wave) < 230000:
                print("processing {}".format(path))
                if ap is None:
                    ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024)
                text = tf.string_to_tensor(transcript).long()
                text_len = torch.LongTensor([len(text)])
                speech = ap.audio_to_mel_spec_tensor(wave).transpose(0, 1)
                speech_len = torch.LongTensor([len(speech)])
                self.cached_text.append(text)
                self.cached_text_lens.append(text_len)
                self.cached_speech.append(speech)
                self.cached_speech_lens.append(speech_len)
                if self.spemb:
                    print("not implemented yet")
                    raise NotImplementedError

    def __getitem__(self, index):
        return self.cached_text[index], \
               self.cached_text_lens[index], \
               self.cached_speech[index], \
               self.cached_speech_lens[index]

    def __len__(self):
        return len(self.cached_text_lens)
