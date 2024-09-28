"""
Example use of the scorer utility to inspect data.
(pre-trained models and cache files with already extracted features are required.)
"""

from Utility.Scorer import TTSScorer
from Utility.path_to_transcript_dicts import *
from Utility.storage_config import PREPROCESSING_DIR

exec_device = "cuda:8"  # ADAPT THIS

lang_id = "eng"
tts_scorer = TTSScorer(path_to_model=None, device=exec_device)
tts_scorer.score(path_to_toucantts_dataset=os.path.join(PREPROCESSING_DIR, "IntegrationTest"), lang_id=lang_id)
tts_scorer.show_samples_with_highest_loss(20)
tts_scorer.remove_samples_with_highest_loss(5)
