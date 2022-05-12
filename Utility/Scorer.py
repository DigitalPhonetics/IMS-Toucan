"""
This module is meant to find potentially problematic samples
in the data you are using. There are two types: The alignment
scorer and the TTS scorer. The alignment scorer can help you
find mispronunciations or errors in the labels. The TTS scorer
can help you find outliers in the audio part of text-audio pairs.
"""

import torch
from tqdm import tqdm


class AlignmentScorer:

    def __init__(self, path_to_aligner_model, device):
        self.aligner = None
        self.path_to_score = dict()
        self.device = device

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
            self.path_to_score[filepaths[index]] = ctc_loss

    def show_samples_with_highest_loss(self, n=-1):
        """
        NaN samples will always be shown.
        To see all samples, pass -1, otherwise n samples will be shown.
        """
        pass


class TTSScorer:

    def __init__(self, path_to_fastspeech_model):
        self.tts = None
        self.path_to_score = dict()

    def score(self, path_to_fastspeech_dataset):
        """
        call this to update the path_to_score dict with scores for this dataset
        """
        datapoints = torch.load(path_to_fastspeech_dataset, map_location='cpu')

    def show_samples_with_highest_loss(self, n=-1):
        """
        NaN samples will always be shown.
        To see all samples, pass -1, otherwise n samples will be shown.
        """
        pass
