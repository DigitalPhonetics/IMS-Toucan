import os
import time

import numpy as np
import torch
from torch.multiprocessing import Manager
from torch.multiprocessing import Process
from torch.utils.data import Dataset
from tqdm import tqdm

from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.DurationCalculator import DurationCalculator
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.EnergyCalculator import EnergyCalculator
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.PitchCalculator import Dio
from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.TacotronDataset import TacotronDataset


class FastSpeechDataset(Dataset):

    def __init__(self,
                 path_to_transcript_dict,
                 acoustic_model,
                 cache_dir,
                 lang,
                 speaker_embedding=False,
                 loading_processes=6,
                 min_len_in_seconds=1,
                 max_len_in_seconds=20,
                 cut_silence=False,
                 reduction_factor=1,
                 device=torch.device("cpu"),
                 rebuild_cache=False,
                 return_language_id = False):

        self.speaker_embedding = speaker_embedding

        if not os.path.exists(os.path.join(cache_dir, "fast_train_cache.pt")) or rebuild_cache:
            if not os.path.exists(os.path.join(cache_dir, "taco_train_cache.pt")) or rebuild_cache:
                TacotronDataset(path_to_transcript_dict=path_to_transcript_dict,
                                cache_dir=cache_dir,
                                lang=lang,
                                speaker_embedding=speaker_embedding,
                                loading_processes=loading_processes,
                                device=device,
                                min_len_in_seconds=min_len_in_seconds,
                                max_len_in_seconds=max_len_in_seconds,
                                cut_silences=cut_silence,
                                rebuild_cache=rebuild_cache)
            datapoints = torch.load(os.path.join(cache_dir, "taco_train_cache.pt"), map_location='cpu')
            # we use the tacotron dataset as basis and augment it to contain the additional information we need for fastspeech.
            if not isinstance(self.datapoints, tuple):  # check for backwards compatibility
                TacotronDataset(path_to_transcript_dict=path_to_transcript_dict,
                                cache_dir=cache_dir,
                                lang=lang,
                                speaker_embedding=speaker_embedding,
                                loading_processes=loading_processes,
                                device=device,
                                min_len_in_seconds=min_len_in_seconds,
                                max_len_in_seconds=max_len_in_seconds,
                                cut_silences=cut_silence,
                                rebuild_cache=True)
                datapoints = torch.load(os.path.join(cache_dir, "taco_train_cache.pt"), map_location='cpu')
            dataset = datapoints[0]
            norm_waves = datapoints[1]

            resource_manager = Manager()
            # build cache
            print("... building dataset cache ...")
            self.datapoints = resource_manager.list()
            # make processes
            datapoint_splits = list()
            norm_wave_splits = list()
            process_list = list()
            for i in range(loading_processes):
                datapoint_splits.append(dataset[i * len(dataset) // loading_processes:(i + 1) * len(dataset) // loading_processes])
                norm_wave_splits.append(norm_waves[i * len(norm_waves) // loading_processes:(i + 1) * len(norm_waves) // loading_processes])
            for index, _ in enumerate(datapoint_splits):
                process_list.append(Process(target=self.cache_builder_process,
                                            args=(datapoint_splits[index],
                                                  norm_wave_splits[index],
                                                  acoustic_model,
                                                  reduction_factor,
                                                  device,
                                                  speaker_embedding),
                                            daemon=True))
                process_list[-1].start()
                time.sleep(5)
            for process in process_list:
                process.join()
            self.datapoints = list(self.datapoints)
            tensored_datapoints = list()
            # we had to turn all of the tensors to numpy arrays to avoid shared memory
            # issues. Now that the multi-processing is over, we can convert them back
            # to tensors to save on conversions in the future.
            print("Converting into convenient format...")
            if self.speaker_embedding:
                for datapoint in tqdm(self.datapoints):
                    tensored_datapoints.append([datapoint[0],
                                                datapoint[1],
                                                datapoint[2],
                                                datapoint[3],
                                                torch.LongTensor(datapoint[4]),  # durations
                                                torch.Tensor(datapoint[5]),  # energy
                                                torch.Tensor(datapoint[6]),  # pitch
                                                datapoint[7]])  # speaker embedding
            else:
                for datapoint in tqdm(self.datapoints):
                    tensored_datapoints.append([datapoint[0],
                                                datapoint[1],
                                                datapoint[2],
                                                datapoint[3],
                                                torch.LongTensor(datapoint[4]),  # durations
                                                torch.Tensor(datapoint[5]),  # energy
                                                torch.Tensor(datapoint[6])])  # pitch
            self.datapoints = tensored_datapoints
            # save to cache
            torch.save(self.datapoints, os.path.join(cache_dir, "fast_train_cache.pt"))
        else:
            # just load the datapoints from cache
            self.datapoints = torch.load(os.path.join(cache_dir, "fast_train_cache.pt"), map_location='cpu')
        print("Prepared {} datapoints.".format(len(self.datapoints)))

    def cache_builder_process(self,
                              datapoint_list,
                              norm_wave_list,
                              acoustic_model,
                              reduction_factor,
                              device,
                              speaker_embedding):
        process_internal_dataset_chunk = list()

        acoustic_model = acoustic_model.to(device)
        dc = DurationCalculator(reduction_factor=reduction_factor)
        dio = Dio(reduction_factor=reduction_factor, fs=16000)
        energy_calc = EnergyCalculator(reduction_factor=reduction_factor, fs=16000)

        for index in tqdm(range(len(datapoint_list))):

            norm_wave = norm_wave_list[index]
            norm_wave_length = torch.LongTensor([len(norm_wave)])

            text = datapoint_list[index][0]
            melspec = datapoint_list[index][2]
            melspec_length = datapoint_list[index][3]

            if not speaker_embedding:
                attention_map = acoustic_model.inference(text_tensor=text.to(device),
                                                         speech_tensor=melspec.to(device),
                                                         use_teacher_forcing=True,
                                                         speaker_embeddings=None)[2]
                cached_duration = dc(attention_map, vis=None)[0].cpu()
            else:
                speaker_embedding = datapoint_list[index][4]
                attention_map = acoustic_model.inference(text_tensor=text.to(device),
                                                         speech_tensor=melspec.to(device),
                                                         use_teacher_forcing=True,
                                                         speaker_embeddings=speaker_embedding.to(device))[2]
                cached_duration = dc(attention_map, vis=None)[0].cpu()
            if np.count_nonzero(cached_duration.numpy() == 0) > 4:
                continue
            cached_energy = energy_calc(input=norm_wave.unsqueeze(0),
                                        input_lengths=norm_wave_length,
                                        feats_lengths=melspec_length,
                                        durations=cached_duration.unsqueeze(0),
                                        durations_lengths=torch.LongTensor([len(cached_duration)]))[0].squeeze(0).cpu().numpy()
            cached_pitch = dio(input=norm_wave.unsqueeze(0),
                               input_lengths=norm_wave_length,
                               feats_lengths=melspec_length,
                               durations=cached_duration.unsqueeze(0),
                               durations_lengths=torch.LongTensor([len(cached_duration)]))[0].squeeze(0).cpu().numpy()
            if not self.speaker_embedding:
                process_internal_dataset_chunk.append([datapoint_list[index][0],
                                                       datapoint_list[index][1],
                                                       datapoint_list[index][2],
                                                       datapoint_list[index][3],
                                                       cached_duration.cpu().numpy(),
                                                       cached_energy,
                                                       cached_pitch])
            else:
                process_internal_dataset_chunk.append([datapoint_list[index][0],
                                                       datapoint_list[index][1],
                                                       datapoint_list[index][2],
                                                       datapoint_list[index][3],
                                                       cached_duration.cpu().numpy(),
                                                       cached_energy,
                                                       cached_pitch,
                                                       datapoint_list[index][4]])
        self.datapoints += process_internal_dataset_chunk

    def __getitem__(self, index):
        if not self.speaker_embedding:
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
