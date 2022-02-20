import torch
import torch.multiprocessing

from TrainingInterfaces.Text_to_Spectrogram.AutoAligner.AlignerDataset import AlignerDataset
from TrainingInterfaces.Text_to_Spectrogram.AutoAligner.autoaligner_train_loop import train_loop as train_aligner
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeechDatasetLanguageID import FastSpeechDataset
from Utility.path_to_transcript_dicts import *


def prepare_aligner_corpus(transcript_dict, corpus_dir, lang, device):
    return AlignerDataset(transcript_dict, cache_dir=corpus_dir, lang=lang, loading_processes=35, cut_silences=True, device=device)


def prepare_fastspeech_corpus(transcript_dict, corpus_dir, lang, ctc_selection=True, fine_tune_aligner=True, use_reconstruction=False):
    """
    create an aligner dataset,
    fine-tune an aligner,
    create a fastspeech dataset,
    return it.

    Skips parts that have been done before.
    """
    if fine_tune_aligner:
        aligner_dir = os.path.join(corpus_dir, "aligner")
        if not os.path.exists(os.path.join(aligner_dir, "aligner.pt")):
            aligner_datapoints = AlignerDataset(transcript_dict, cache_dir=corpus_dir, lang=lang)
            train_aligner(train_dataset=aligner_datapoints,
                          device=torch.device("cuda"),
                          save_directory=aligner_dir,
                          steps=(len(aligner_datapoints) * 5) // 32,
                          batch_size=32,
                          path_to_checkpoint="Models/Aligner/aligner.pt",
                          fine_tune=True,
                          debug_img_path=aligner_dir,
                          resume=False,
                          use_reconstruction=use_reconstruction)
    else:
        aligner_dir = "Models/Aligner/"
    ds = FastSpeechDataset(transcript_dict,
                           acoustic_checkpoint_path=os.path.join(aligner_dir, "aligner.pt"),
                           cache_dir=corpus_dir,
                           device=torch.device("cuda"),
                           lang=lang,
                           ctc_selection=ctc_selection)
    ds.fix_repeating_phones()
    return 1
