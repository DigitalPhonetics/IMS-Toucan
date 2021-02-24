import os

import soundfile as sf
import torch
from torch.utils.data import Dataset

from FastSpeech.DurationCalculator import DurationCalculator
from FastSpeech.EnergyCalculator import EnergyCalculator
from FastSpeech.PitchCalculator import Dio
from PreprocessingForTTS.ProcessAudio import AudioPreprocessor
from PreprocessingForTTS.ProcessText import TextFrontend
from TransformerTTS.TransformerTTS import build_transformertts_model


class FastSpeechDataset(Dataset):

    def __init__(self, path_to_transcript_dict, acoustic_model_name, device=torch.device("cpu"), spemb=False,
                 train=True):
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
        acoustic_model = build_transformertts_model(model_name=acoustic_model_name)
        dc = DurationCalculator()
        dio = Dio()  ###################
        energy_calc = EnergyCalculator()  #######################
        # build cache
        print("... building dataset cache ...")
        self.cached_text = list()
        self.cached_text_lens = list()
        self.cached_speech = list()
        self.cached_speech_lens = list()
        self.cached_energy = list()
        self.cached_pitch = list()
        self.cached_durations = list()
        for path in key_list:
            transcript = self.path_to_transcript_dict[path]
            wave, sr = sf.read(os.path.join("Corpora/CSS10/", path))
            if 50000 < len(wave) < 230000:
                print("processing {}".format(path))
                if ap is None:
                    ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024)
                if self.spemb:
                    print("not implemented yet")
                    raise NotImplementedError
                else:
                    self.cached_durations.append(dc(acoustic_model.inference(text=self.cached_text[-1],
                                                                             speech=self.cached_speech[-1],
                                                                             use_teacher_forcing=True,
                                                                             spembs=None)[2]))
                self.cached_text.append(tf.string_to_tensor(transcript).long())
                self.cached_text_lens.append(torch.LongTensor([len(self.cached_text[-1])]))
                self.cached_speech.append(ap.audio_to_mel_spec_tensor(wave).transpose(0, 1))
                self.cached_speech_lens.append(torch.LongTensor([len(self.cached_speech[-1])]))

    def __getitem__(self, index):
        return self.cached_text[index], \
               self.cached_text_lens[index], \
               self.cached_speech[index], \
               self.cached_speech_lens[index], \
               self.cached_durations[index], \
               self.cached_energy[index], \
               self.cached_pitch[index]

    def __len__(self):
        return len(self.cached_text_lens)
