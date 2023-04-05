"""
Example use of the scorer utility to inspect data.

(pre-)trained models and already cache files with extracted features are required.
"""

import torch

from Utility.Scorer import AlignmentScorer
from Utility.Scorer import TTSScorer
from Utility.corpus_preparation import prepare_fastspeech_corpus
from Utility.path_to_transcript_dicts import *
from Utility.silence_removal import make_sielce_cleaned_versions
from Utility.storage_config import MODELS_DIR
from Utility.storage_config import PREPROCESSING_DIR

exec_device = "cuda" if torch.cuda.is_available() else "cpu"

alignment_scorer = AlignmentScorer(path_to_aligner_model=os.path.join(MODELS_DIR, "Aligner", "aligner.pt"), device=exec_device)
alignment_scorer.score(path_to_aligner_dataset=os.path.join(PREPROCESSING_DIR, "IntegrationTest", "aligner_train_cache.pt"))
alignment_scorer.show_samples_with_highest_loss(20)

tts_scorer = TTSScorer(path_to_model=os.path.join(MODELS_DIR, "ToucanTTS_Meta", "best.pt"), device=exec_device)
tts_scorer.score(path_to_toucantts_dataset=os.path.join(PREPROCESSING_DIR, "IntegrationTest/"), lang_id="en")
tts_scorer.show_samples_with_highest_loss(20)
tts_scorer.remove_samples_with_highest_loss(5)

train_sets = list()
train_sets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_integration_test(),
                                            corpus_dir=os.path.join(PREPROCESSING_DIR, "IntegrationTest"),
                                            lang="en"))
make_sielce_cleaned_versions(train_sets=train_sets)
