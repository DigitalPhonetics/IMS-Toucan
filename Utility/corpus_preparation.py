import torch
import torch.multiprocessing

from TrainingInterfaces.Text_to_Spectrogram.AutoAligner.AlignerDataset import AlignerDataset
from TrainingInterfaces.Text_to_Spectrogram.AutoAligner.autoaligner_train_loop import train_loop as train_aligner
from TrainingInterfaces.Text_to_Spectrogram.ToucanTTS.TTSDataset import TTSDataset
from Utility.path_to_transcript_dicts import *
from Utility.storage_config import MODELS_DIR


def prepare_aligner_corpus(transcript_dict, corpus_dir, lang, device, do_loudnorm=True, cut_silences=True, phone_input=False, speaker_embedding_func=None):
    return AlignerDataset(transcript_dict,
                          cache_dir=corpus_dir,
                          lang=lang,
                          loading_processes=4,  # this can be increased for massive clusters, but the overheads that are introduced are kind of not really worth it
                          cut_silences=cut_silences,
                          do_loudnorm=do_loudnorm,
                          device=device,
                          phone_input=phone_input,
                          speaker_embedding_func=speaker_embedding_func)


def prepare_tts_corpus(transcript_dict,
                       corpus_dir,
                       lang,
                       ctc_selection=True,  # heuristically removes some samples which might be problematic.
                       # For small datasets it's best to turn this off and instead inspect the data with the scorer, if there are any issues.
                       fine_tune_aligner=True,
                       use_reconstruction=True,
                       phone_input=False,
                       save_imgs=False):
    """
    create an aligner dataset,
    fine-tune an aligner,
    create a TTS dataset,
    return it.

    Automatically skips parts that have been done before.
    """
    if not os.path.exists(os.path.join(corpus_dir, "tts_train_cache.pt")):
        if fine_tune_aligner:
            aligner_dir = os.path.join(corpus_dir, "Aligner")
            aligner_loc = os.path.join(corpus_dir, "Aligner", "aligner.pt")

            if not os.path.exists(os.path.join(aligner_dir, "aligner.pt")):
                aligner_datapoints = prepare_aligner_corpus(transcript_dict, corpus_dir=corpus_dir, lang=lang, phone_input=phone_input, device=torch.device("cuda"))
                if os.path.exists(os.path.join(MODELS_DIR, "Aligner", "aligner.pt")):
                    train_aligner(train_dataset=aligner_datapoints,
                                  device=torch.device("cuda"),
                                  save_directory=aligner_dir,
                                  steps=len(aligner_datapoints),  # relatively good finetuning heuristic
                                  batch_size=32 if len(aligner_datapoints) > 32 else len(aligner_datapoints) // 2,
                                  path_to_checkpoint=os.path.join(MODELS_DIR, "Aligner", "aligner.pt"),
                                  fine_tune=True,
                                  debug_img_path=aligner_dir,
                                  resume=False,
                                  use_reconstruction=use_reconstruction)
                else:
                    train_aligner(train_dataset=aligner_datapoints,
                                  device=torch.device("cuda"),
                                  save_directory=aligner_dir,
                                  steps=len(aligner_datapoints) * 2,  # relatively good heuristic
                                  batch_size=32 if len(aligner_datapoints) > 32 else len(aligner_datapoints) // 2,
                                  path_to_checkpoint=None,
                                  fine_tune=False,
                                  debug_img_path=aligner_dir,
                                  resume=False,
                                  use_reconstruction=use_reconstruction)
        else:
            aligner_loc = os.path.join(MODELS_DIR, "Aligner", "aligner.pt")
    else:
        aligner_loc = None
    return TTSDataset(transcript_dict,
                      acoustic_checkpoint_path=aligner_loc,
                      cache_dir=corpus_dir,
                      device=torch.device("cuda"),
                      lang=lang,
                      ctc_selection=ctc_selection,
                      save_imgs=save_imgs)
