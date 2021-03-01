import json
import os
from multiprocessing import Process, Manager

import soundfile as sf
import torch
from torch.utils.data import Dataset

from FastSpeech2.DurationCalculator import DurationCalculator
from FastSpeech2.EnergyCalculator import EnergyCalculator
from FastSpeech2.PitchCalculator import Dio
from PreprocessingForTTS.ProcessAudio import AudioPreprocessor
from PreprocessingForTTS.ProcessText import TextFrontend
from TransformerTTS.TransformerTTS import build_reference_transformer_tts_model


class FastSpeechDataset(Dataset):

    def __init__(self,
                 path_to_transcript_dict,
                 acoustic_model_name,
                 device=torch.device("cpu"),
                 spemb=False,
                 train=True,
                 loading_processes=2,
                 save=True,
                 load=False):
        if not load:
            ressource_manager = Manager()
            self.path_to_transcript_dict = path_to_transcript_dict
            if type(train) is str:
                key_list = list(self.path_to_transcript_dict.keys())[:1]
            else:
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
                    Process(target=self.cache_builder_process, args=(key_split, acoustic_model_name, spemb),
                            daemon=True))
                process_list[-1].start()
            for process in process_list:
                process.join()
            self.datapoints = list(self.datapoints)
            if save:
                # save to json so we can rebuild cache quickly
                if train:
                    with open(os.path.join("Corpora", "CSS10", "fast_train_cache.json"), 'w') as fp:
                        json.dump(self.datapoints, fp)
                else:
                    with open(os.path.join("Corpora", "CSS10", "fast_valid_cache.json"), 'w') as fp:
                        json.dump(self.datapoints, fp)
        else:
            # just load the datapoints
            if train:
                with open(os.path.join("Corpora", "CSS10", "fast_train_cache.json"), 'r') as fp:
                    self.datapoints = json.load(fp)
            else:
                with open(os.path.join("Corpora", "CSS10", "fast_valid_cache.json"), 'r') as fp:
                    self.datapoints = json.load(fp)

    def cache_builder_process(self, path_list, acoustic_model_name, spemb):
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
        acoustic_model = build_reference_transformer_tts_model(model_name=acoustic_model_name)
        dc = DurationCalculator()
        dio = Dio()
        energy_calc = EnergyCalculator()
        for path in path_list:
            transcript = self.path_to_transcript_dict[path]
            wave, _ = sf.read(os.path.join("Corpora/CSS10/", path))
            if 50000 < len(wave) < 230000:
                print("processing {}".format(path))
                norm_wave = ap.audio_to_wave_tensor(audio=wave, normalize=True, mulaw=False)
                norm_wave_length = torch.LongTensor([len(norm_wave)])
                melspec = ap.audio_to_mel_spec_tensor(norm_wave, normalize=False).transpose(0, 1)
                melspec_length = torch.LongTensor([len(melspec)])
                text = tf.string_to_tensor(transcript).long()
                cached_text = tf.string_to_tensor(transcript).numpy().tolist()
                cached_text_lens = len(cached_text)
                cached_speech = ap.audio_to_mel_spec_tensor(wave).transpose(0, 1).numpy().tolist()
                cached_speech_lens = len(cached_speech)
                if not spemb:
                    cached_durations = dc(acoustic_model.inference(text=text,
                                                                   speech=melspec,
                                                                   use_teacher_forcing=True,
                                                                   spembs=None)[2])[0]
                else:
                    raise NotImplementedError
                cached_energy = energy_calc(input=norm_wave.unsqueeze(0),
                                            input_lengths=norm_wave_length,
                                            feats_lengths=melspec_length,
                                            durations=cached_durations.unsqueeze(0),
                                            durations_lengths=len(cached_durations))[0].squeeze(0)
                cached_pitch = dio(input=norm_wave.unsqueeze(0),
                                   input_lengths=norm_wave_length,
                                   feats_lengths=melspec_length,
                                   durations=cached_durations.unsqueeze(0),
                                   durations_lengths=len(cached_durations))[0].squeeze(0)
                self.datapoints.append(
                    [cached_text,
                     cached_text_lens,
                     cached_speech,
                     cached_speech_lens,
                     cached_durations.numpy().tolist(),
                     cached_energy.numpy().tolist(),
                     cached_pitch.numpy().tolist()])
                if self.spemb:
                    print("not implemented yet")
                    raise NotImplementedError

    def __getitem__(self, index):
        return self.datapoints[index][0], \
               self.datapoints[index][1], \
               self.datapoints[index][2], \
               self.datapoints[index][3], \
               self.datapoints[index][4], \
               self.datapoints[index][5], \
               self.datapoints[index][6]

    def __len__(self):
        return len(self.datapoints)
