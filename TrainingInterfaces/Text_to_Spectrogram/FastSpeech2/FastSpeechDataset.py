import os
import statistics

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from Preprocessing.ArticulatoryCombinedTextFrontend import ArticulatoryCombinedTextFrontend
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
                 loading_processes=40,
                 min_len_in_seconds=1,
                 max_len_in_seconds=20,
                 cut_silence=False,
                 reduction_factor=1,
                 device=torch.device("cpu"),
                 rebuild_cache=False,
                 ctc_selection=True,
                 save_imgs=False):
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
            if save_imgs:
                vis_dir = os.path.join(cache_dir, "duration_vis")
                os.makedirs(vis_dir, exist_ok=True)
            pros_cond_ext = ProsodicConditionExtractor(sr=16000, device=device)

            orig_texts = list()
            tf_no_pauses = ArticulatoryCombinedTextFrontend(language=lang, use_word_boundaries=False)
            tf_with_pauses = ArticulatoryCombinedTextFrontend(language=lang, use_word_boundaries=True)
            for index in tqdm(range(len(dataset))):
                text = dataset[index][0]
                for graphemes in path_to_transcript_dict.values():
                    if tf_no_pauses.string_to_tensor(graphemes) == text:
                        orig_texts.append(tf_with_pauses.string_to_tensor(graphemes))

            for index in tqdm(range(len(dataset))):
                norm_wave = norm_waves[index]
                norm_wave_length = torch.LongTensor([len(norm_wave)])

                if len(norm_wave) / 16000 < min_len_in_seconds and ctc_selection:
                    continue

                text = orig_texts[index]
                melspec = dataset[index][2]
                melspec_length = dataset[index][3]

                alignment_path, ctc_loss = acoustic_model.inference(mel=melspec.to(device),
                                                                    tokens=text.to(device),
                                                                    save_img_for_debug=os.path.join(vis_dir, f"{index}.png") if save_imgs else None,
                                                                    return_ctc=True)

                cached_duration = dc(torch.LongTensor(alignment_path), vis=None).cpu()

                # now we check for all silence labels whether they have a duration of less than 10 frames.
                # If so, we remove it from the text, update the text length and recompute alignment.

                silence_phoneme = torch.tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

                pop_indexes = list()
                for sequence_index, phoneme in enumerate(text):
                    if phoneme == silence_phoneme:
                        if cached_duration[sequence_index] < 10:  # silence too short to be substantial, let's exclude it.
                            pop_indexes.append(sequence_index)
                if pop_indexes:
                    for pop_index in sorted(pop_indexes, reverse=True):
                        text.pop(pop_index)
                    alignment_path, ctc_loss = acoustic_model.inference(mel=melspec.to(device),
                                                                        tokens=text.to(device),
                                                                        save_img_for_debug=os.path.join(vis_dir, f"{index}_fixed.png") if save_imgs else None,
                                                                        return_ctc=True)
                    cached_duration = dc(torch.LongTensor(alignment_path), vis=None).cpu()

                text_len = len(text)

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

                self.datapoints.append([text,
                                        text_len,
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

            if ctc_selection:
                # now we can filter out some bad datapoints based on the CTC scores we collected
                mean_ctc = sum(self.ctc_losses) / len(self.ctc_losses)
                std_dev = statistics.stdev(self.ctc_losses)
                threshold = mean_ctc + std_dev
                for index in range(len(self.ctc_losses), 0, -1):
                    if self.ctc_losses[index - 1] > threshold:
                        self.datapoints.pop(index - 1)
                        print(
                            f"Removing datapoint {index - 1}, because the CTC loss is one standard deviation higher than the mean. \n ctc: {round(self.ctc_losses[index - 1], 4)} vs. mean: {round(mean_ctc, 4)}")

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
