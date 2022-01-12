import os
import statistics

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from Preprocessing.ProsodicConditionExtractor import ProsodicConditionExtractor
from TrainingInterfaces.Text_to_Spectrogram.AutoAligner.Aligner import Aligner
from TrainingInterfaces.Text_to_Spectrogram.AutoAligner.AlignerDataset import AlignerDataset
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.DurationCalculator import DurationCalculator
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.EnergyCalculator import EnergyCalculator
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.PitchCalculator import Dio


class FastSpeechDataset(Dataset):

    def __init__(self,
                 path_to_transcript_dict,
                 acoustic_checkpoint_path,
                 cache_dir,
                 lang,
                 loading_processes=6,
                 min_len_in_seconds=1,
                 max_len_in_seconds=20,
                 cut_silence=False,
                 reduction_factor=1,
                 device=torch.device("cpu"),
                 rebuild_cache=False):
        os.makedirs(cache_dir, exist_ok=True)
        if not os.path.exists(os.path.join(cache_dir, "fast_train_cache.pt")) or rebuild_cache:
            if not os.path.exists(os.path.join(cache_dir, "aligner_train_cache.pt")) or rebuild_cache:
                AlignerDataset(path_to_transcript_dict=path_to_transcript_dict,
                               cache_dir=cache_dir,
                               lang=lang,
                               loading_processes=loading_processes,
                               min_len_in_seconds=min_len_in_seconds,
                               max_len_in_seconds=max_len_in_seconds,
                               cut_silences=cut_silence,
                               rebuild_cache=rebuild_cache)
            datapoints = torch.load(os.path.join(cache_dir, "aligner_train_cache.pt"), map_location='cpu')
            # we use the aligner dataset as basis and augment it to contain the additional information we need for fastspeech.
            if not isinstance(datapoints, tuple):  # check for backwards compatibility
                AlignerDataset(path_to_transcript_dict=path_to_transcript_dict,
                               cache_dir=cache_dir,
                               lang=lang,
                               loading_processes=loading_processes,
                               min_len_in_seconds=min_len_in_seconds,
                               max_len_in_seconds=max_len_in_seconds,
                               cut_silences=cut_silence,
                               rebuild_cache=True)
                datapoints = torch.load(os.path.join(cache_dir, "aligner_train_cache.pt"), map_location='cpu')
            dataset = datapoints[0]
            norm_waves = datapoints[1]

            # build cache
            print("... building dataset cache ...")
            self.datapoints = list()
            self.ctc_losses = list()

            acoustic_model = Aligner()
            acoustic_model.load_state_dict(torch.load(acoustic_checkpoint_path, map_location='cpu')["asr_model"])

            # ==========================================
            # actual creation of datapoints starts here
            # ==========================================

            acoustic_model = acoustic_model.to(device)
            dio = Dio(reduction_factor=reduction_factor, fs=16000)
            energy_calc = EnergyCalculator(reduction_factor=reduction_factor, fs=16000)
            dc = DurationCalculator(reduction_factor=reduction_factor)
            vis_dir = os.path.join(cache_dir, "duration_vis")
            os.makedirs(vis_dir, exist_ok=True)
            pros_cond_ext = ProsodicConditionExtractor(sr=16000, device=device)

            for index in tqdm(range(len(dataset))):
                norm_wave = norm_waves[index]
                norm_wave_length = torch.LongTensor([len(norm_wave)])

                text = dataset[index][0]
                melspec = dataset[index][2]
                melspec_length = dataset[index][3]

                alignment_path, ctc_loss = acoustic_model.inference(mel=melspec.to(device),
                                                                    tokens=text.to(device),
                                                                    save_img_for_debug=os.path.join(vis_dir, f"{index}.png"),
                                                                    return_ctc=True)

                cached_duration = dc(torch.LongTensor(alignment_path), vis=None).cpu()

                cached_energy = energy_calc(input_waves=norm_wave.unsqueeze(0),
                                            input_waves_lengths=norm_wave_length,
                                            feats_lengths=melspec_length,
                                            durations=cached_duration.unsqueeze(0),
                                            durations_lengths=torch.LongTensor([len(cached_duration)]))[0].squeeze(0).cpu()

                cached_pitch = dio(input_waves=norm_wave.unsqueeze(0),
                                   input_waves_lengths=norm_wave_length,
                                   feats_lengths=melspec_length,
                                   durations=cached_duration.unsqueeze(0),
                                   durations_lengths=torch.LongTensor([len(cached_duration)]))[0].squeeze(0).cpu()

                try:
                    prosodic_condition = pros_cond_ext.extract_condition_from_reference_wave(norm_wave, already_normalized=True).cpu()
                except RuntimeError:
                    # if there is an audio without any voiced segments whatsoever we have to skip it.
                    continue

                self.datapoints.append([dataset[index][0],
                                        dataset[index][1],
                                        dataset[index][2],
                                        dataset[index][3],
                                        cached_duration.cpu(),
                                        cached_energy,
                                        cached_pitch,
                                        prosodic_condition])
                self.ctc_losses.append(ctc_loss)

            # =============================
            # done with datapoint creation
            # =============================

            # now we can filter out some bad datapoints based on the CTC scores we collected
            mean_ctc = sum(self.ctc_losses) / len(self.ctc_losses)
            std_dev = statistics.stdev(self.ctc_losses)
            threshold = mean_ctc + std_dev
            for index in range(len(self.ctc_losses), 0, -1):
                if self.ctc_losses[index - 1] > threshold:
                    self.datapoints.pop(index - 1)
                    print(
                        f"Removing datapoint {index - 1}, because the CTC loss indicates there's something wrong with it. "
                        f"Maybe the label is partially incorrect. ctc: {round(self.ctc_losses[index - 1], 4)} vs. mean: {round(mean_ctc, 4)}")

            # save to cache
            if len(self.datapoints) > 0:
                torch.save(self.datapoints, os.path.join(cache_dir, "fast_train_cache.pt"))
            else:
                import sys
                print("No datapoints were prepared! Exiting...")
                sys.exit()
        else:
            # just load the datapoints from cache
            self.datapoints = torch.load(os.path.join(cache_dir, "fast_train_cache.pt"), map_location='cpu')

        print("Prepared {} datapoints.".format(len(self.datapoints)))

    def __getitem__(self, index):
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
