"""
Example use of the scorer utility to inspect data.

(pre-)trained models and already cache files with extracted features are required.
"""

from Utility.Scorer import AlignmentScorer
from Utility.Scorer import TTSScorer

alignment_scorer = AlignmentScorer(path_to_aligner_model="Models/Aligner/aligner.pt", device="cpu")
alignment_scorer.score(path_to_aligner_dataset="Corpora/IntegrationTest/aligner_train_cache.pt")
alignment_scorer.show_samples_with_highest_loss(50)

alignment_scorer = TTSScorer(path_to_fastspeech_model="Models/FastSpeech2_IntegrationTest/best.pt", device="cpu")
alignment_scorer.score(path_to_fastspeech_dataset="Corpora/IntegrationTest/fast_train_cache.pt")
alignment_scorer.show_samples_with_highest_loss(50)
