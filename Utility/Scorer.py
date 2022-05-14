"""
This module is meant to find potentially problematic samples
in the data you are using. There are two types: The alignment
scorer and the TTS scorer. The alignment scorer can help you
find mispronunciations or errors in the labels. The TTS scorer
can help you find outliers in the audio part of text-audio pairs.
"""

import math

import torch
from tqdm import tqdm

from Preprocessing.TextFrontend import get_language_id
from TrainingInterfaces.Text_to_Spectrogram.AutoAligner.Aligner import Aligner
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2


class AlignmentScorer:

    def __init__(self, path_to_aligner_model, device):
        self.path_to_score = dict()
        self.device = device
        self.nans = list()
        self.aligner = Aligner()
        self.aligner.load_state_dict(torch.load(path_to_aligner_model, map_location='cpu')["asr_model"])
        self.aligner.to(self.device)

    def score(self, path_to_aligner_dataset):
        """
        call this to update the path_to_score dict with scores for this dataset
        """
        datapoints = torch.load(path_to_aligner_dataset, map_location='cpu')
        dataset = datapoints[0]
        filepaths = datapoints[3]
        for index in tqdm(range(len(dataset))):
            text = dataset[index][0]
            melspec = dataset[index][2]
            text_without_word_boundaries = list()
            for phoneme_index, vector in enumerate(text):
                if vector[19] == 0:
                    text_without_word_boundaries.append(vector.numpy().tolist())
            matrix_without_word_boundaries = torch.Tensor(text_without_word_boundaries)
            _, ctc_loss = self.aligner.inference(mel=melspec.to(self.device),
                                                 tokens=matrix_without_word_boundaries.to(self.device),
                                                 save_img_for_debug=None,
                                                 return_ctc=True)
            if math.isnan(ctc_loss):
                self.nans.append(filepaths[index])
            self.path_to_score[filepaths[index]] = ctc_loss
        if len(self.nans) > 0:
            print("The following filepaths had an infinite loss:")
            for path in self.nans:
                print(path)

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


class TTSScorer:

    def __init__(self, path_to_fastspeech_model, device):
        self.path_to_score = dict()
        self.device = device
        self.nans = list()
        self.tts = FastSpeech2()
        weights = torch.load(path_to_fastspeech_model, map_location='cpu')["model"]
        try:
            self.tts.load_state_dict(weights)
        except RuntimeError:
            try:
                self.tts = FastSpeech2(lang_embs=None)
                self.tts.load_state_dict(weights)
            except RuntimeError:
                self.tts = FastSpeech2(lang_embs=None, utt_embed_dim=None)
                self.tts.load_state_dict(weights)
        self.tts.to(self.device)

    def score(self, path_to_fastspeech_dataset, lang_id):
        """
        call this to update the path_to_score dict with scores for this dataset
        """
        datapoints = torch.load(path_to_fastspeech_dataset, map_location='cpu')
        for index in tqdm(range(len(datapoints))):
            text, text_len, spec, spec_len, duration, energy, pitch, embed, filepath = datapoints[index]
            loss = self.tts(text_tensors=text.unsqueeze(0).to(self.device),
                            text_lengths=text_len.to(self.device),
                            gold_speech=spec.unsqueeze(0).to(self.device),
                            speech_lengths=spec_len.to(self.device),
                            gold_durations=duration.unsqueeze(0).to(self.device),
                            gold_pitch=pitch.unsqueeze(0).to(self.device),
                            gold_energy=energy.unsqueeze(0).to(self.device),
                            utterance_embedding=embed.unsqueeze(0).to(self.device),
                            lang_ids=get_language_id(lang_id).unsqueeze(0).to(self.device),
                            return_mels=False)
            if torch.isnan(loss):
                self.nans.append(filepath)
            self.path_to_score[filepath] = loss.cpu().item()
        if len(self.nans) > 0:
            print("The following filepaths had an infinite loss:")
            for path in self.nans:
                print(path)

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
