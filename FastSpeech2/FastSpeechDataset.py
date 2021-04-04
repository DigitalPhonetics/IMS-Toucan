import json
import os
from multiprocessing import Process, Manager

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm

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
                 spemb=False,
                 train=True,
                 loading_processes=6,
                 cache_dir=os.path.join("Corpora", "CSS10_DE"),
                 lang="de",
                 min_len_in_seconds=1,
                 max_len_in_seconds=20,
                 reduction_factor=1,
                 device=torch.device("cpu"),
                 rebuild_cache=False):
        self.spemb = spemb
        if ((not os.path.exists(os.path.join(cache_dir, "fast_train_cache.json"))) and train) or (
                (not os.path.exists(os.path.join(cache_dir, "fast_valid_cache.json"))) and (not train)) or \
                rebuild_cache:
            if not os.path.isdir(os.path.join(cache_dir, "durations_visualization")):
                os.makedirs(os.path.join(cache_dir, "durations_visualization"))
            ressource_manager = Manager()
            self.path_to_transcript_dict = path_to_transcript_dict
            if type(train) is str:
                key_list = list(self.path_to_transcript_dict.keys())[:1]
            else:
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
                            args=(
                                key_split, acoustic_model_name, spemb, lang, min_len_in_seconds, max_len_in_seconds,
                                reduction_factor, device, cache_dir),
                            daemon=True))
                process_list[-1].start()
            for process in process_list:
                process.join()
            self.datapoints = list(self.datapoints)
            # save to json so we can rebuild cache quickly
            if train:
                with open(os.path.join(cache_dir, "fast_train_cache.json"), 'w') as fp:
                    json.dump(self.datapoints, fp)
            else:
                with open(os.path.join(cache_dir, "fast_valid_cache.json"), 'w') as fp:
                    json.dump(self.datapoints, fp)
        else:
            # just load the datapoints
            if train:
                with open(os.path.join(cache_dir, "fast_train_cache.json"), 'r') as fp:
                    self.datapoints = json.load(fp)
            else:
                with open(os.path.join(cache_dir, "fast_valid_cache.json"), 'r') as fp:
                    self.datapoints = json.load(fp)
        print("Prepared {} datapoints.".format(len(self.datapoints)))

    def cache_builder_process(self,
                              path_list,
                              acoustic_model_name,
                              spemb,
                              lang,
                              min_len,
                              max_len,
                              reduction_factor,
                              device,
                              cache_dir):
        tf = TextFrontend(language=lang,
                          use_panphon_vectors=False,
                          use_word_boundaries=False,
                          use_explicit_eos=False)
        _, sr = sf.read(path_list[0])
        if spemb:
            wav2mel = torch.jit.load("Models/Use/SpeakerEmbedding/wav2mel.pt")
            dvector = torch.jit.load("Models/Use/SpeakerEmbedding/dvector-step250000.pt").eval()
        ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024)
        acoustic_model = build_reference_transformer_tts_model(model_name=acoustic_model_name).to(device)
        dc = DurationCalculator(reduction_factor=reduction_factor)
        dio = Dio(reduction_factor=reduction_factor)
        energy_calc = EnergyCalculator(reduction_factor=reduction_factor)
        for index, path in tqdm(enumerate(path_list)):
            transcript = self.path_to_transcript_dict[path]
            with open(path, "rb") as audio_file:
                wave, sr = sf.read(audio_file)
            if min_len <= len(wave) / sr <= max_len:
                # print("Processing {} out of {}.".format(index, len(path_list)))
                norm_wave = ap.audio_to_wave_tensor(audio=wave, normalize=True, mulaw=False)
                norm_wave_length = torch.LongTensor([len(norm_wave)])
                melspec = ap.audio_to_mel_spec_tensor(norm_wave, normalize=False).transpose(0, 1)
                melspec_length = torch.LongTensor([len(melspec)])
                text = tf.string_to_tensor(transcript).long()
                cached_text = tf.string_to_tensor(transcript).squeeze(0).numpy().tolist()
                cached_text_lens = len(cached_text)
                cached_speech = ap.audio_to_mel_spec_tensor(wave).transpose(0, 1).numpy().tolist()
                cached_speech_lens = len(cached_speech)
                if not spemb:
                    os.path.join(cache_dir, "durations_visualization")
                    cached_durations = dc(acoustic_model.inference(text=text.squeeze(0).to(device),
                                                                   speech=melspec.to(device),
                                                                   use_teacher_forcing=True,
                                                                   spembs=None)[2],
                                          vis=os.path.join(cache_dir, "durations_visualization",
                                                           path.split("/")[-1].rstrip(".wav") + ".png"))[0].cpu()
                else:
                    wav_tensor, sample_rate = torchaudio.load(path)
                    mel_tensor = wav2mel(wav_tensor, sample_rate)
                    cached_spemb = dvector.embed_utterance(mel_tensor)
                    cached_durations = dc(acoustic_model.inference(text=text.squeeze(0).to(device),
                                                                   speech=melspec.to(device),
                                                                   use_teacher_forcing=True,
                                                                   spembs=cached_spemb.to(device))[2],
                                          vis=os.path.join(cache_dir, "durations_visualization",
                                                           ".".join(path.split(".")[:-1]) + ".png"))[0].cpu()
                cached_energy = energy_calc(input=norm_wave.unsqueeze(0),
                                            input_lengths=norm_wave_length,
                                            feats_lengths=melspec_length,
                                            durations=cached_durations.unsqueeze(0),
                                            durations_lengths=torch.LongTensor([len(cached_durations)]))[0].squeeze(0)
                cached_pitch = dio(input=norm_wave.unsqueeze(0),
                                   input_lengths=norm_wave_length,
                                   feats_lengths=melspec_length,
                                   durations=cached_durations.unsqueeze(0),
                                   durations_lengths=torch.LongTensor([len(cached_durations)]))[0].squeeze(0)
                self.datapoints.append(
                    [cached_text,
                     cached_text_lens,
                     cached_speech,
                     cached_speech_lens,
                     cached_durations.numpy().tolist(),
                     cached_energy.numpy().tolist(),
                     cached_pitch.numpy().tolist()])
                if self.spemb:
                    self.datapoints.append(
                        [cached_text,
                         cached_text_lens,
                         cached_speech,
                         cached_speech_lens,
                         cached_durations.numpy().tolist(),
                         cached_energy.numpy().tolist(),
                         cached_pitch.numpy().tolist(),
                         cached_spemb.detach().numpy().tolist()])

    def __getitem__(self, index):
        if not self.spemb:
            return self.datapoints[index][0], \
                   self.datapoints[index][1], \
                   self.datapoints[index][2], \
                   self.datapoints[index][3], \
                   self.datapoints[index][4], \
                   self.datapoints[index][5], \
                   self.datapoints[index][6]
        else:
            return self.datapoints[index][0], \
                   self.datapoints[index][1], \
                   self.datapoints[index][2], \
                   self.datapoints[index][3], \
                   self.datapoints[index][4], \
                   self.datapoints[index][5], \
                   self.datapoints[index][6], \
                   self.datapoints[index][7]

    def __len__(self):
        return len(self.datapoints)
