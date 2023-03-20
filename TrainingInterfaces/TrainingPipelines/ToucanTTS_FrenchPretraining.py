import time

import torch
import wandb
from torch.utils.data import ConcatDataset

from TrainingInterfaces.Text_to_Spectrogram.ToucanTTS.ToucanTTS import ToucanTTS
from TrainingInterfaces.Text_to_Spectrogram.ToucanTTS.toucantts_train_loop_arbiter import train_loop
from Utility.blizzard_pretraining_path_to_transcript import *
from Utility.corpus_preparation import prepare_fastspeech_corpus
from Utility.path_to_transcript_dicts import *
from Utility.storage_config import MODELS_DIR
from Utility.storage_config import PREPROCESSING_DIR


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume, use_wandb, wandb_resume_id):
    if gpu_id == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")

    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
        device = torch.device("cuda")

    torch.manual_seed(131714)
    random.seed(131714)
    torch.random.manual_seed(131714)

    print("Preparing")

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join(MODELS_DIR, "ToucanTTS_blizzard_pretraining")
    os.makedirs(save_dir, exist_ok=True)

    train_sets = list()

    train_sets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_blizzard2023_ad(),
                                                corpus_dir=os.path.join(PREPROCESSING_DIR, "blizzard2023ad"),
                                                lang="fr"))

    train_sets.append(prepare_fastspeech_corpus(transcript_dict=read_pfc(),
                                                corpus_dir=os.path.join(PREPROCESSING_DIR, "pfc"),
                                                lang="fr"))

    chunk_count = 5
    mls_chunks = split_dictionary(read_mls(), split_n=chunk_count)
    for index in range(chunk_count):
        train_sets.append(prepare_fastspeech_corpus(transcript_dict=mls_chunks[index],
                                                    corpus_dir=os.path.join(PREPROCESSING_DIR, f"mls_french_female_chunk_{index}"),
                                                    lang="fr"))

    train_sets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_blizzard2023_neb(),
                                                corpus_dir=os.path.join(PREPROCESSING_DIR, "blizzard2023neb"),
                                                lang="fr"))

    train_sets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_siwis_subset(),
                                                corpus_dir=os.path.join(PREPROCESSING_DIR, "siwis"),
                                                lang="fr"))

    model = ToucanTTS()
    if use_wandb:
        wandb.init(
            name=f"{__name__.split('.')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}" if wandb_resume_id is None else None,
            id=wandb_resume_id,  # this is None if not specified in the command line arguments.
            resume="must" if wandb_resume_id is not None else None)
    print("Training model")
    train_loop(net=model,
               datasets=[ConcatDataset(train_sets)],
               device=device,
               save_directory=save_dir,
               eval_lang="fr",
               path_to_checkpoint=resume_checkpoint,
               path_to_embed_model=os.path.join(MODELS_DIR, "Embedding", "embedding_function.pt"),
               fine_tune=finetune,
               resume=resume,
               warmup_steps=8000,
               postnet_start_steps=9000,
               steps=120000,
               use_wandb=use_wandb)
    if use_wandb:
        wandb.finish()


def split_dictionary(input_dict, split_n):
    res = []
    new_dict = {}
    elements_per_dict = (len(input_dict.keys()) // split_n) + 1
    for k, v in input_dict.items():
        if len(new_dict) < elements_per_dict:
            new_dict[k] = v
        else:
            res.append(new_dict)
            new_dict = {k: v}
    res.append(new_dict)
    return res


def create_cache(pttd, cachedir, lang):
    prepare_fastspeech_corpus(transcript_dict=pttd,
                              corpus_dir=cachedir,
                              lang=lang)
