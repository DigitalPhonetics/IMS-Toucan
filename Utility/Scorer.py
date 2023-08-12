"""
This module is meant to find potentially problematic samples
in the data you are using. There are two types: The alignment
scorer and the TTS scorer. The alignment scorer can help you
find mispronunciations or errors in the labels. The TTS scorer
can help you find outliers in the audio part of text-audio pairs.
"""

import math
import os

import torch
from tqdm import tqdm

from Preprocessing.TextFrontend import get_language_id
from Preprocessing.articulatory_features import get_feature_to_index_lookup
from TrainingInterfaces.Spectrogram_to_Embedding.StyleEmbedding import StyleEmbedding
from TrainingInterfaces.Text_to_Spectrogram.AutoAligner.Aligner import Aligner
from TrainingInterfaces.Text_to_Spectrogram.ToucanTTS.ToucanTTS import ToucanTTS
from Utility.corpus_preparation import prepare_fastspeech_corpus
from Utility.storage_config import MODELS_DIR
from Utility.utils import get_speakerid_from_path_all, get_speakerid_from_path


class AlignmentScorer:

    def __init__(self, path_to_aligner_model, device):
        self.path_to_score = dict()
        self.path_to_id = dict()
        self.device = device
        self.nans = list()
        self.nan_indexes = list()
        self.aligner = Aligner()
        self.aligner.load_state_dict(torch.load(path_to_aligner_model, map_location='cpu')["asr_model"])
        self.aligner.to(self.device)
        self.datapoints = None
        self.path_to_aligner_dataset = None

    def score(self, path_to_aligner_dataset):
        """
        call this to update the path_to_score dict with scores for this dataset
        """
        datapoints = torch.load(path_to_aligner_dataset, map_location='cpu')
        self.path_to_aligner_dataset = path_to_aligner_dataset
        self.datapoints = datapoints[0]
        self.norm_waves = datapoints[1]
        self.speaker_embeddings = datapoints[2]
        self.filepaths = datapoints[3]
        dataset = datapoints[0]
        filepaths = datapoints[3]
        self.nans = list()
        self.nan_indexes = list()
        self.path_to_score = dict()
        self.path_to_id = dict()
        for index in tqdm(range(len(dataset))):
            text = dataset[index][0]
            melspec = dataset[index][2]
            text_without_word_boundaries = list()
            for phoneme_index, vector in enumerate(text):
                if vector[get_feature_to_index_lookup()["word-boundary"]] == 0:
                    text_without_word_boundaries.append(vector.numpy().tolist())
            matrix_without_word_boundaries = torch.Tensor(text_without_word_boundaries)
            _, ctc_loss = self.aligner.inference(mel=melspec.to(self.device),
                                                 tokens=matrix_without_word_boundaries.to(self.device),
                                                 save_img_for_debug=None,
                                                 return_ctc=True)
            if math.isnan(ctc_loss):
                self.nans.append(filepaths[index])
                self.nan_indexes.append(index)
            self.path_to_score[filepaths[index]] = ctc_loss
            self.path_to_id[filepaths[index]] = index
        if len(self.nans) > 0:
            print("The following filepaths had an infinite loss:")
            for path in self.nans:
                print(path)
        self.save_scores()

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

    def save_scores(self):
        if self.path_to_score is None:
            print("Please run the scoring first.")
        else:
            torch.save((self.path_to_score, self.path_to_id, self.nan_indexes), 
                       os.path.join(os.path.dirname(self.path_to_aligner_dataset), 'alignment_scores.pt'))

    def remove_samples_with_highest_loss(self, path_to_aligner_dataset, n=10):
        if self.datapoints is None:
            self.path_to_aligner_dataset = path_to_aligner_dataset
            datapoints = torch.load(self.path_to_aligner_dataset, map_location='cpu')
            self.datapoints = datapoints[0]
            self.norm_waves = datapoints[1]
            self.speaker_embeddings = datapoints[2]
            self.filepaths = datapoints[3]
            try:
                alignment_scores = torch.load(os.path.join(os.path.dirname(self.path_to_aligner_dataset), 'alignment_scores.pt'), map_location='cpu')
                self.path_to_score = alignment_scores[0]
                self.path_to_id = alignment_scores[1]
                self.nan_indexes = alignment_scores[2]
            except FileNotFoundError:
                print("Please run the scoring first.")
                return
        remove_ids = list()
        remove_ids.extend(self.nan_indexes)
        for index, path in enumerate(sorted(self.path_to_score, key=self.path_to_score.get, reverse=True)):
            if index < n:
                remove_ids.append(self.path_to_id[path])
        for remove_id in sorted(remove_ids, reverse=True):
            self.datapoints.pop(remove_id)
            self.norm_waves.pop(remove_id)
            self.speaker_embeddings.pop(remove_id)
            self.filepaths.pop(remove_id)
        torch.save((self.datapoints, self.norm_waves, self.speaker_embeddings, self.filepaths),
                self.path_to_aligner_dataset)
        print("Dataset updated!")

class TTSScorer:

    def __init__(self,
                 path_to_model,
                 device,
                 path_to_embedding_checkpoint=None,
                 static_speaker_embed=False,
                 ):
        self.device = device
        self.path_to_score = dict()
        self.path_to_id = dict()
        self.nans = list()
        self.nan_indexes = list()
        self.tts = ToucanTTS()
        checkpoint = torch.load(path_to_model, map_location='cpu')
        weights = checkpoint["model"]
        try:
            self.tts.load_state_dict(weights)
        except RuntimeError:
            try:
                self.tts = ToucanTTS(lang_embs=None)
                self.tts.load_state_dict(weights)
            except RuntimeError:
                try:
                    self.tts = ToucanTTS(lang_embs=None, utt_embed_dim=None)
                    self.tts.load_state_dict(weights)
                except RuntimeError:
                    self.tts = ToucanTTS(lang_embs=None, utt_embed_dim=512, static_speaker_embed=True)
                    self.tts.load_state_dict(weights)
        if path_to_embedding_checkpoint is not None:
            self.style_embedding_function = StyleEmbedding().to(device)
            check_dict = torch.load(path_to_embedding_checkpoint, map_location=device)
            self.style_embedding_function.load_state_dict(check_dict["style_emb_func"])
            self.style_embedding_function.to(device)
        else:
            self.style_embedding_function = None
        self.tts.to(self.device)
        self.static_speaker_embed = static_speaker_embed
        self.nans_removed = False
        self.current_dset = None

    def score(self, path_to_toucantts_dataset, lang_id):
        """
        call this to update the path_to_score dict with scores for this dataset
        """
        dataset = prepare_fastspeech_corpus(dict(), path_to_toucantts_dataset, lang_id)
        self.current_dset = dataset
        self.nans = list()
        self.nan_indexes = list()
        self.path_to_score = dict()
        self.path_to_id = dict()
        if self.static_speaker_embed:
            with open("/mount/arbeitsdaten/synthesis/bottts/IMS-Toucan/Corpora/librittsr/libri_speakers.txt") as f:
                libri_speakers = sorted([int(line.rstrip()) for line in f])
        for index in tqdm(range(len(dataset.datapoints))):
            text, text_len, spec, spec_len, duration, energy, pitch, embed, filepath = dataset.datapoints[index]
            if self.style_embedding_function is not None:
                style_embedding = self.style_embedding_function(batch_of_spectrograms=spec.unsqueeze(0).to(self.device),
                                                                batch_of_spectrogram_lengths=spec_len.unsqueeze(0).to(self.device))
            else:
                style_embedding = None
            if self.static_speaker_embed:
                speaker_id = torch.LongTensor([get_speakerid_from_path(filepath, libri_speakers)]).to(self.device)
            else:
                speaker_id = None
            try:
                l1_loss, \
                duration_loss, \
                pitch_loss, \
                energy_loss, \
                glow_loss = self.tts(text_tensors=text.unsqueeze(0).to(self.device),
                                            text_lengths=text_len.to(self.device),
                                            gold_speech=spec.unsqueeze(0).to(self.device),
                                            speech_lengths=spec_len.to(self.device),
                                            gold_durations=duration.unsqueeze(0).to(self.device),
                                            gold_pitch=pitch.unsqueeze(0).to(self.device),
                                            gold_energy=energy.unsqueeze(0).to(self.device),
                                            utterance_embedding=style_embedding.to(self.device) if style_embedding is not None else None,
                                            speaker_id=speaker_id,
                                            lang_ids=get_language_id(lang_id).unsqueeze(0).to(self.device),
                                            return_mels=False,
                                            run_glow=False)
                loss = l1_loss + duration_loss + pitch_loss + energy_loss
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
