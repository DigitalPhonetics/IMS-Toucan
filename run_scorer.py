"""
Example use of the scorer utility to inspect data.
(pre-trained models and already cache files with extracted features are required.)"""

from Utility.Scorer import TTSScorer
from Utility.path_to_transcript_dicts import *
from Utility.storage_config import MODELS_DIR
from Utility.storage_config import PREPROCESSING_DIR

exec_device = "cuda:8"

lang_id = "fon"
tts_scorer = TTSScorer(path_to_model=os.path.join(MODELS_DIR, "ToucanTTS_Massive", "best.pt"), device=exec_device)
tts_scorer.score(path_to_toucantts_dataset=os.path.join(PREPROCESSING_DIR, "african_voices_fon_alf"), lang_id=lang_id)
tts_scorer.show_samples_with_highest_loss(20)
tts_scorer.remove_samples_with_highest_loss(5)

lang_id = "hau"
tts_scorer = TTSScorer(path_to_model=os.path.join(MODELS_DIR, "ToucanTTS_Massive", "best.pt"), device=exec_device)
tts_scorer.score(path_to_toucantts_dataset=os.path.join(PREPROCESSING_DIR, "african_voices_hausa_cmv"), lang_id=lang_id)
tts_scorer.show_samples_with_highest_loss(20)
tts_scorer.remove_samples_with_highest_loss(5)

lang_id = "lbb"
tts_scorer = TTSScorer(path_to_model=os.path.join(MODELS_DIR, "ToucanTTS_Massive", "best.pt"), device=exec_device)
tts_scorer.score(path_to_toucantts_dataset=os.path.join(PREPROCESSING_DIR, "african_voices_ibibio_lst"), lang_id=lang_id)
tts_scorer.show_samples_with_highest_loss(20)
tts_scorer.remove_samples_with_highest_loss(5)

lang_id = "kik"
tts_scorer = TTSScorer(path_to_model=os.path.join(MODELS_DIR, "ToucanTTS_Massive", "best.pt"), device=exec_device)
tts_scorer.score(path_to_toucantts_dataset=os.path.join(PREPROCESSING_DIR, "african_voices_kikuyu_opb"), lang_id=lang_id)
tts_scorer.show_samples_with_highest_loss(20)
tts_scorer.remove_samples_with_highest_loss(5)

lang_id = "lin"
tts_scorer = TTSScorer(path_to_model=os.path.join(MODELS_DIR, "ToucanTTS_Massive", "best.pt"), device=exec_device)
tts_scorer.score(path_to_toucantts_dataset=os.path.join(PREPROCESSING_DIR, "african_voices_lingala_opb"), lang_id=lang_id)
tts_scorer.show_samples_with_highest_loss(20)
tts_scorer.remove_samples_with_highest_loss(5)

lang_id = "lug"
tts_scorer = TTSScorer(path_to_model=os.path.join(MODELS_DIR, "ToucanTTS_Massive", "best.pt"), device=exec_device)
tts_scorer.score(path_to_toucantts_dataset=os.path.join(PREPROCESSING_DIR, "african_voices_ganda_cmv"), lang_id=lang_id)
tts_scorer.show_samples_with_highest_loss(20)
tts_scorer.remove_samples_with_highest_loss(5)

lang_id = "luo"
tts_scorer = TTSScorer(path_to_model=os.path.join(MODELS_DIR, "ToucanTTS_Massive", "best.pt"), device=exec_device)
tts_scorer.score(path_to_toucantts_dataset=os.path.join(PREPROCESSING_DIR, "african_voices_luo_afv"), lang_id=lang_id)
tts_scorer.show_samples_with_highest_loss(20)
tts_scorer.remove_samples_with_highest_loss(5)

lang_id = "luo"
tts_scorer = TTSScorer(path_to_model=os.path.join(MODELS_DIR, "ToucanTTS_Massive", "best.pt"), device=exec_device)
tts_scorer.score(path_to_toucantts_dataset=os.path.join(PREPROCESSING_DIR, "african_voices_luo_opb"), lang_id=lang_id)
tts_scorer.show_samples_with_highest_loss(20)
tts_scorer.remove_samples_with_highest_loss(5)

lang_id = "swh"
tts_scorer = TTSScorer(path_to_model=os.path.join(MODELS_DIR, "ToucanTTS_Massive", "best.pt"), device=exec_device)
tts_scorer.score(path_to_toucantts_dataset=os.path.join(PREPROCESSING_DIR, "african_voices_swahili_llsti"), lang_id=lang_id)
tts_scorer.show_samples_with_highest_loss(20)
tts_scorer.remove_samples_with_highest_loss(5)

lang_id = "sxb"
tts_scorer = TTSScorer(path_to_model=os.path.join(MODELS_DIR, "ToucanTTS_Massive", "best.pt"), device=exec_device)
tts_scorer.score(path_to_toucantts_dataset=os.path.join(PREPROCESSING_DIR, "african_voices_suba_afv"), lang_id=lang_id)
tts_scorer.show_samples_with_highest_loss(20)
tts_scorer.remove_samples_with_highest_loss(5)

lang_id = "wol"
tts_scorer = TTSScorer(path_to_model=os.path.join(MODELS_DIR, "ToucanTTS_Massive", "best.pt"), device=exec_device)
tts_scorer.score(path_to_toucantts_dataset=os.path.join(PREPROCESSING_DIR, "african_voices_wolof_alf"), lang_id=lang_id)
tts_scorer.show_samples_with_highest_loss(20)
tts_scorer.remove_samples_with_highest_loss(5)

lang_id = "yor"
tts_scorer = TTSScorer(path_to_model=os.path.join(MODELS_DIR, "ToucanTTS_Massive", "best.pt"), device=exec_device)
tts_scorer.score(path_to_toucantts_dataset=os.path.join(PREPROCESSING_DIR, "african_voices_yoruba_opb"), lang_id=lang_id)
tts_scorer.show_samples_with_highest_loss(20)
tts_scorer.remove_samples_with_highest_loss(5)

lang_id = "nya"
tts_scorer = TTSScorer(path_to_model=os.path.join(MODELS_DIR, "ToucanTTS_Massive", "best.pt"), device=exec_device)
tts_scorer.score(path_to_toucantts_dataset=os.path.join(PREPROCESSING_DIR, "zambezi_voice_nyanja"), lang_id=lang_id)
tts_scorer.show_samples_with_highest_loss(20)
tts_scorer.remove_samples_with_highest_loss(5)

lang_id = "loz"
tts_scorer = TTSScorer(path_to_model=os.path.join(MODELS_DIR, "ToucanTTS_Massive", "best.pt"), device=exec_device)
tts_scorer.score(path_to_toucantts_dataset=os.path.join(PREPROCESSING_DIR, "zambezi_voice_lozi"), lang_id=lang_id)
tts_scorer.show_samples_with_highest_loss(20)
tts_scorer.remove_samples_with_highest_loss(5)

lang_id = "toi"
tts_scorer = TTSScorer(path_to_model=os.path.join(MODELS_DIR, "ToucanTTS_Massive", "best.pt"), device=exec_device)
tts_scorer.score(path_to_toucantts_dataset=os.path.join(PREPROCESSING_DIR, "zambezi_voice_tonga"), lang_id=lang_id)
tts_scorer.show_samples_with_highest_loss(20)
tts_scorer.remove_samples_with_highest_loss(5)
