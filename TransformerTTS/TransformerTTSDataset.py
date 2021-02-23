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
            self.key_list = list(self.path_to_transcript_dict.keys())[:-100]
        else:
            self.key_list = list(self.path_to_transcript_dict.keys())[-100:]
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
        self.ap = None

    def __getitem__(self, index):
        transcript = self.path_to_transcript_dict[self.key_list[index]]
        path = self.key_list[index]
        wave, sr = sf.read(os.path.join("Corpora/CSS10/", path))
        if self.ap is None:
            self.ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024)
        text = self.tf.string_to_tensor(transcript).long()
        text_len = torch.LongTensor([len(text)])
        speech = self.ap.audio_to_mel_spec_tensor(wave).transpose(0, 1)
        speech_len = torch.LongTensor([len(speech)])
        if self.spemb:
            print("not implemented yet")
            raise NotImplementedError
        return text, text_len, speech, speech_len

    def __len__(self):
        return len(self.key_list)
