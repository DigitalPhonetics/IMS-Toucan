import torch
import torch.multiprocessing

from Architectures.Aligner.CodecAlignerDataset import CodecAlignerDataset
from Architectures.Aligner.autoaligner_train_loop import train_loop as train_aligner
from Architectures.ToucanTTS.TTSDataset import TTSDataset
from Utility.path_to_transcript_dicts import *
from Utility.storage_config import MODELS_DIR


def prepare_aligner_corpus(transcript_dict, corpus_dir, lang, device, phone_input=False):
    return CodecAlignerDataset(transcript_dict,
                               cache_dir=corpus_dir,
                               lang=lang,
                               loading_processes=4,  # this can be increased for massive clusters, but the overheads that are introduced are kind of not really worth it
                               device=device,
                               phone_input=phone_input)


def prepare_tts_corpus(transcript_dict,
                       corpus_dir,
                       lang,
                       # For small datasets it's best to turn this off and instead inspect the data with the scorer, if there are any issues.
                       fine_tune_aligner=True,
                       use_reconstruction=True,
                       phone_input=False,
                       save_imgs=False,
                       gpu_count=1,
                       rank=0):
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

            if not os.path.exists(os.path.join(corpus_dir, "aligner_train_cache.pt")):
                prepare_aligner_corpus(transcript_dict, corpus_dir=corpus_dir, lang=lang, phone_input=phone_input, device=torch.device("cuda"))

            if not os.path.exists(os.path.join(aligner_dir, "aligner.pt")):
                aligner_datapoints = prepare_aligner_corpus(transcript_dict, corpus_dir=corpus_dir, lang=lang, phone_input=phone_input, device=torch.device("cuda"))
                if os.path.exists(os.path.join(MODELS_DIR, "Aligner", "aligner.pt")):
                    train_aligner(train_dataset=aligner_datapoints,
                                  device=torch.device("cuda"),
                                  save_directory=aligner_dir,
                                  steps=min(len(aligner_datapoints) // 2, 10000),  # relatively good finetuning heuristic
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
                                  steps=len(aligner_datapoints) // 2,  # relatively good heuristic
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
                      save_imgs=save_imgs,
                      gpu_count=gpu_count,
                      rank=rank)
