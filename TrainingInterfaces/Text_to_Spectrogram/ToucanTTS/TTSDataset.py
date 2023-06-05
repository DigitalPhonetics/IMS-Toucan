import os

import soundfile as sf
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from Preprocessing.TextFrontend import get_language_id
from Preprocessing.articulatory_features import get_feature_to_index_lookup
from TrainingInterfaces.Text_to_Spectrogram.AutoAligner.Aligner import Aligner
from TrainingInterfaces.Text_to_Spectrogram.AutoAligner.AlignerDataset import AlignerDataset
from TrainingInterfaces.Text_to_Spectrogram.ToucanTTS.DurationCalculator import DurationCalculator
from TrainingInterfaces.Text_to_Spectrogram.ToucanTTS.EnergyCalculator import EnergyCalculator
from TrainingInterfaces.Text_to_Spectrogram.ToucanTTS.PitchCalculator import Parselmouth


class TTSDataset(Dataset):

    def __init__(self,
                 path_to_transcript_dict,
                 acoustic_checkpoint_path,
                 cache_dir,
                 lang,
                 loading_processes=os.cpu_count() if os.cpu_count() is not None else 30,
                 min_len_in_seconds=1,
                 max_len_in_seconds=15,
                 cut_silence=False,
                 do_loudnorm=True,
                 reduction_factor=1,
                 device=torch.device("cpu"),
                 rebuild_cache=False,
                 save_imgs=False):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        if not os.path.exists(os.path.join(cache_dir, "tts_train_cache.pt")) or rebuild_cache:
            if not os.path.exists(os.path.join(cache_dir, "aligner_train_cache.pt")) or rebuild_cache:
                AlignerDataset(path_to_transcript_dict=path_to_transcript_dict,
                               cache_dir=cache_dir,
                               lang=lang,
                               loading_processes=loading_processes,
                               min_len_in_seconds=min_len_in_seconds,
                               max_len_in_seconds=max_len_in_seconds,
                               cut_silences=cut_silence,
                               do_loudnorm=do_loudnorm,
                               rebuild_cache=rebuild_cache,
                               device=device)
            aligner_datapoint_feature_dump_list = torch.load(os.path.join(cache_dir, "aligner_train_cache.pt"), map_location='cpu')
            # we use the aligner dataset as basis and augment it to contain the additional information we need for fastspeech.

            print("... building dataset cache ...")
            self.datapoints = list()

            acoustic_model = Aligner()
            acoustic_model.load_state_dict(torch.load(acoustic_checkpoint_path, map_location='cpu')["asr_model"])

            # ==========================================
            # actual creation of datapoints starts here
            # ==========================================

            acoustic_model = acoustic_model.to(device).eval()
            initial_file = aligner_datapoint_feature_dump_list[0]
            _, _, filepath = torch.load(initial_file, map_location='cpu')

            _, _orig_sr = sf.read(filepath)
            parsel = Parselmouth(reduction_factor=reduction_factor, fs=_orig_sr)
            energy_calc = EnergyCalculator(reduction_factor=reduction_factor, fs=_orig_sr)
            dc = DurationCalculator(reduction_factor=reduction_factor)
            vis_dir = os.path.join(cache_dir, "duration_vis")
            os.makedirs(vis_dir, exist_ok=True)

            for index in tqdm(range(len(aligner_datapoint_feature_dump_list))):
                datapoint, _, filepath = torch.load(aligner_datapoint_feature_dump_list[index], map_location='cpu')
                raw_wave, sr = sf.read(filepath)
                if _orig_sr != sr:
                    print(f"Not all files have the same sampling rate! Please fix and re-run.  -- triggered by {filepath}")

                norm_wave_length = torch.LongTensor([len(raw_wave)])

                text = datapoint[0]
                melspec = datapoint[2]
                melspec_length = datapoint[3]

                # We deal with the word boundaries by having 2 versions of text: with and without word boundaries.
                # We note the index of word boundaries and insert durations of 0 afterwards
                text_without_word_boundaries = list()
                indexes_of_word_boundaries = list()
                for phoneme_index, vector in enumerate(text):
                    if vector[get_feature_to_index_lookup()["word-boundary"]] == 0:
                        text_without_word_boundaries.append(vector.numpy().tolist())
                    else:
                        indexes_of_word_boundaries.append(phoneme_index)
                matrix_without_word_boundaries = torch.Tensor(text_without_word_boundaries)

                alignment_path = acoustic_model.inference(mel=melspec.to(device),
                                                          tokens=matrix_without_word_boundaries.to(device),
                                                          save_img_for_debug=os.path.join(vis_dir, f"{index}.png") if save_imgs else None,
                                                          return_ctc=False)

                cached_duration = dc(torch.LongTensor(alignment_path), vis=None).cpu()

                last_vec = None
                for phoneme_index, vec in enumerate(text):
                    if last_vec is not None:
                        if torch.equal(last_vec, vec):
                            # we found a case of repeating phonemes!
                            # now we must repair their durations by giving the first one 3/5 of their sum and the second one 2/5 (i.e. the rest)
                            total_dur = cached_duration[phoneme_index - 1] + cached_duration[phoneme_index]
                            new_dur_1 = int((total_dur / 5) * 3)
                            cached_duration[phoneme_index - 1] = new_dur_1
                            cached_duration[phoneme_index] = total_dur - new_dur_1
                    last_vec = vec

                # adding 0 durations for the word boundaries at the indexes we noted down previously
                new_size = cached_duration.size(0) + len(indexes_of_word_boundaries)  # Calculate the size of the new tensor
                new_tensor = torch.zeros(new_size, dtype=cached_duration.dtype)  # Create a new tensor with the desired size
                inserted_index = 0
                for i, idx in enumerate(indexes_of_word_boundaries):
                    new_tensor[inserted_index:idx] = cached_duration[inserted_index:idx]
                    new_tensor[idx] = torch.LongTensor([0])
                    inserted_index = idx + 1
                new_tensor[inserted_index:] = cached_duration[indexes_of_word_boundaries[-1] + 1:]  # Copy the remaining values after the last index

                # the following lines are the previous solution, but concatenations are slow compared to creating a new tensor with the correct size and filling it.
                # for index_of_word_boundary in indexes_of_word_boundaries:
                #    cached_duration = torch.cat([cached_duration[:index_of_word_boundary],
                #                                 torch.LongTensor([0]),  # insert a 0 duration wherever there is a word boundary
                #                                 cached_duration[index_of_word_boundary:]])

                input_wave = torch.Tensor(raw_wave).unsqueeze(0)
                cached_energy = energy_calc(input_waves=input_wave,
                                            input_waves_lengths=norm_wave_length,
                                            feats_lengths=melspec_length,
                                            text=text,
                                            durations=cached_duration.unsqueeze(0),
                                            durations_lengths=torch.LongTensor([len(cached_duration)]))[0].squeeze(0).cpu().float()

                cached_pitch = parsel(input_waves=input_wave,
                                      input_waves_lengths=norm_wave_length,
                                      feats_lengths=melspec_length,
                                      text=text,
                                      durations=cached_duration.unsqueeze(0),
                                      durations_lengths=torch.LongTensor([len(cached_duration)]))[0].squeeze(0).cpu().float()

                self.datapoints.append([datapoint[0],
                                        datapoint[1],
                                        datapoint[2],
                                        datapoint[3],
                                        cached_duration.cpu(),
                                        cached_energy,
                                        cached_pitch,
                                        None,  # this used to be the prosodic condition, but is now deprecated
                                        filepath])

            # =============================
            # done with datapoint creation
            # =============================

            # save to cache
            os.makedirs(os.path.join(cache_dir, f"tts_datapoints/"), exist_ok=True)
            if len(self.datapoints) > 0:
                self.datapoint_feature_dump_list = list()
                for index, datapoint in enumerate(self.datapoints):
                    torch.save(datapoint,
                               os.path.join(cache_dir, f"tts_datapoints/tts_datapoint_{index}.pt"))
                    self.datapoint_feature_dump_list.append(os.path.join(cache_dir, f"tts_datapoints/tts_datapoint_{index}.pt"))
                torch.save(self.datapoint_feature_dump_list,
                           os.path.join(cache_dir, "tts_train_cache.pt"))
                del self.datapoints
            else:
                import sys
                print("No datapoints were prepared! Exiting...")
                sys.exit()
        else:
            # just load the datapoints from cache
            self.datapoint_feature_dump_list = torch.load(os.path.join(cache_dir, "tts_train_cache.pt"), map_location='cpu')

        self.cache_dir = cache_dir
        self.language_id = get_language_id(lang)
        print(f"Prepared a FastSpeech dataset with {len(self.datapoint_feature_dump_list)} datapoints in {cache_dir}.")

    def __getitem__(self, index):
        datapoint = torch.load(self.datapoint_feature_dump_list[index], map_location='cpu')
        return datapoint[0], \
               datapoint[1], \
               datapoint[2], \
               datapoint[3], \
               datapoint[4], \
               datapoint[5], \
               datapoint[6], \
               datapoint[7], \
               self.language_id

    def __len__(self):
        return len(self.datapoint_feature_dump_list)

    def remove_samples(self, list_of_samples_to_remove):
        for remove_id in sorted(list_of_samples_to_remove, reverse=True):
            self.datapoints.pop(remove_id)
        torch.save(self.datapoints, os.path.join(self.cache_dir, "tts_train_cache.pt"))
        print("Dataset updated!")
