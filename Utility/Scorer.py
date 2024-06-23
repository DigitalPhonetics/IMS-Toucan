"""
This module is meant to find potentially problematic samples
in the data you are using. There are two types: The alignment
scorer and the TTS scorer. The alignment scorer can help you
find mispronunciations or errors in the labels. The TTS scorer
can help you find outliers in the audio part of text-audio pairs.
"""

import math
import statistics

import torch
import torch.multiprocessing
from tqdm import tqdm

from Architectures.ToucanTTS.ToucanTTS import ToucanTTS
from Preprocessing.AudioPreprocessor import AudioPreprocessor
from Preprocessing.EnCodecAudioPreprocessor import CodecAudioPreprocessor
from Utility.corpus_preparation import prepare_tts_corpus


class TTSScorer:

    def __init__(self,
                 path_to_model,
                 device,
                 ):
        self.device = device
        self.path_to_score = dict()
        self.path_to_id = dict()
        self.nans = list()
        self.nan_indexes = list()
        self.tts = ToucanTTS()
        checkpoint = torch.load(path_to_model, map_location='cpu')
        weights = checkpoint["model"]
        self.tts.load_state_dict(weights)
        self.tts.to(self.device)
        self.nans_removed = False
        self.current_dset = None
        self.ap = CodecAudioPreprocessor(input_sr=-1, device=device)
        self.spec_extractor = AudioPreprocessor(input_sr=16000, output_sr=16000, device=device)

    def score(self, path_to_toucantts_dataset, lang_id):
        """
        call this to update the path_to_score dict with scores for this dataset
        """
        dataset = prepare_tts_corpus(dict(), path_to_toucantts_dataset, lang_id)
        self.current_dset = dataset
        self.nans = list()
        self.nan_indexes = list()
        self.path_to_score = dict()
        self.path_to_id = dict()
        _ = dataset[0]
        for index in tqdm(range(len(dataset.datapoints))):
            datapoint = dataset.datapoints[index]
            text_tensors = datapoint[0].to(self.device).unsqueeze(0).float()
            text_lengths = datapoint[1].squeeze().to(self.device).unsqueeze(0)
            speech_indexes = datapoint[2]
            speech_lengths = datapoint[3].squeeze().to(self.device).unsqueeze(0)
            gold_durations = datapoint[4].to(self.device).unsqueeze(0)
            gold_pitch = datapoint[6].to(self.device).unsqueeze(0)  # mind the switched order
            gold_energy = datapoint[5].to(self.device).unsqueeze(0)  # mind the switched order
            lang_ids = dataset.language_id.to(self.device)
            filepath = datapoint[8]
            with torch.inference_mode():
                wave = self.ap.indexes_to_audio(speech_indexes.int().to(self.device)).detach()
                mel = self.spec_extractor.audio_to_mel_spec_tensor(wave, explicit_sampling_rate=16000).transpose(0, 1).detach().cpu()
            gold_speech_sample = mel.clone().to(self.device).unsqueeze(0)

            utterance_embedding = datapoint[7].unsqueeze(0).to(self.device)
            try:
                regression_loss, _, duration_loss, pitch_loss, energy_loss = self.tts(text_tensors=text_tensors,
                                                                                      text_lengths=text_lengths,
                                                                                      gold_speech=gold_speech_sample,
                                                                                      speech_lengths=speech_lengths,
                                                                                      gold_durations=gold_durations,
                                                                                      gold_pitch=gold_pitch,
                                                                                      gold_energy=gold_energy,
                                                                                      utterance_embedding=utterance_embedding,
                                                                                      lang_ids=lang_ids,
                                                                                      return_feats=False,
                                                                                      run_stochastic=False)
                loss = regression_loss  # + duration_loss + pitch_loss + energy_loss  # we omit the stochastic loss
            except TypeError:
                loss = torch.tensor(torch.nan)
            if torch.isnan(loss):
                self.nans.append(filepath)
                self.nan_indexes.append(index)
            self.path_to_score[filepath] = loss.cpu().item()
            self.path_to_id[filepath] = index
        if len(self.nans) > 0:
            print("NaNs detected during scoring!")
            for path in self.nans:
                print(path)
            print("\n\n")
        self.nans_removed = False

    def show_samples_with_highest_loss(self, n=-1):
        """
        NaN samples will always be shown.
        To see all samples, pass -1, otherwise n samples will be shown.
        """
        if len(self.nans) > 0:
            print("The following filepaths had an infinite loss:")
            for path in self.nans:
                print(path)
            print("\n\n")

        for index, path in enumerate(sorted(self.path_to_score, key=self.path_to_score.get, reverse=True)):
            if index < n or n == -1:
                print(f"Loss: {round(self.path_to_score[path], 3)} - Path: {path}")
        print("\n\n")

    def remove_samples_with_highest_loss(self, n=10):
        if self.current_dset is None:
            print("Please run the scoring first.")
        else:
            if self.nans_removed:
                print("Indexes are no longer accurate. Please re-run the scoring. \n\n"
                      "This function also removes NaNs, so if you want to remove the NaN samples and the n samples "
                      "with the highest loss, only call this function.")
            else:
                remove_ids = list()
                remove_ids.extend(self.nan_indexes)
                for index, path in enumerate(sorted(self.path_to_score, key=self.path_to_score.get, reverse=True)):
                    if index < n:
                        remove_ids.append(self.path_to_id[path])
                self.current_dset.remove_samples(remove_ids)
                self.nans_removed = True

    def remove_samples_with_loss_three_std_devs_higher_than_avg(self):
        if self.current_dset is None:
            print("Please run the scoring first.")
        else:
            if self.nans_removed:
                print("Indexes are no longer accurate. Please re-run the scoring. \n\n"
                      "This function also removes NaNs, so if you want to remove the NaN samples and the outliers, only call this one here.")
            else:
                remove_ids = list()
                remove_ids.extend(self.nan_indexes)
                scores_without_nans = [value for value in list(self.path_to_score.values()) if not math.isnan(value)]
                avg = statistics.mean(scores_without_nans)
                std = statistics.stdev(scores_without_nans)
                thresh = avg + (3 * std)
                for path in self.path_to_score:
                    if not math.isnan(self.path_to_score[path]):
                        if self.path_to_score[path] > thresh:  # we found an outlier!
                            remove_ids.append(self.path_to_id[path])
                print(f"removing {len(remove_ids)} outliers!")
                self.current_dset.remove_samples(remove_ids)
                self.nans_removed = True

    def remove_nans(self):
        if self.nans_removed:
            print("NaNs have already been removed!")
        else:
            if self.current_dset is None:
                print("Please run the scoring first to find NaNs.")
            else:
                if len(self.nans) > 0:
                    print("The following filepaths had an infinite loss and are being removed from the dataset cache:")
                    for path in self.nans:
                        print(path)
                    self.current_dset.remove_samples(self.nan_indexes)
                    self.nans_removed = True
                else:
                    print("No NaNs detected in this dataset.")
