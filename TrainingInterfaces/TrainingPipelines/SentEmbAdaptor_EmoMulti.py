import time

import torch
import wandb
from torch.utils.data import ConcatDataset

from TrainingInterfaces.Text_to_Embedding.SentenceEmbeddingAdaptor import SentenceEmbeddingAdaptor
from TrainingInterfaces.Text_to_Embedding.sent_emb_adaptor_train_loop import train_loop
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
        device = torch.device(f"cuda")

    torch.manual_seed(131714)
    random.seed(131714)
    torch.random.manual_seed(131714)

    print("Preparing")

    name = "SentEmbAdaptor_01_EmoMulti_emoBERTcls"

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join(MODELS_DIR, name)
    os.makedirs(save_dir, exist_ok=True)

    datasets = list()

    datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_EmoV_DB(),
                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "emovdb"),
                                          lang="en",
                                          save_imgs=False))
    
    datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_CREMA_D(),
                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "cremad"),
                                          lang="en",
                                          save_imgs=False))

    datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_RAVDESS(),
                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "ravdess"),
                                          lang="en",
                                          save_imgs=False))
    
    datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_ESDS(),
                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "esds"),
                                          lang="en",
                                          save_imgs=False))
    
    train_set = ConcatDataset(datasets)
    
    if "laser" in name:
        embed_type = "laser"
        sent_embed_dim = 1024
    if "lealla" in name:
        embed_type = "lealla"
        sent_embed_dim = 192
    if "para" in name:
        embed_type = "para"
        sent_embed_dim = 768
    if "mpnet" in name:
        embed_type = "mpnet"
        sent_embed_dim = 768
    if "bertcls" in name:
        embed_type = "bertcls"
        sent_embed_dim = 768
    if "bertlm" in name:
        embed_type = "bertlm"
        sent_embed_dim = 768
    if "emoBERTcls" in name:
        embed_type = "emoBERTcls"
        sent_embed_dim = 768

    print(f'Loading sentence embeddings from {os.path.join(PREPROCESSING_DIR, "Yelp", f"emotion_prompts_large_sent_embs_{embed_type}.pt")}')
    sent_embs = torch.load(os.path.join(PREPROCESSING_DIR, "Yelp", f"emotion_prompts_large_sent_embs_{embed_type}.pt"), map_location='cpu')
    
    model = SentenceEmbeddingAdaptor(sent_embed_dim=sent_embed_dim, utt_embed_dim=64)

    if use_wandb:
        wandb.init(
            name=f"{name}_{time.strftime('%Y%m%d-%H%M%S')}" if wandb_resume_id is None else None,
            id=wandb_resume_id,  # this is None if not specified in the command line arguments.
            resume="must" if wandb_resume_id is not None else None)
    print("Training model")
    train_loop(net=model,
               train_dataset=train_set,
               device=device,
               save_directory=save_dir,
               batch_size=16,
               lr=0.001,
               warmup_steps=4000,
               path_to_checkpoint=resume_checkpoint,
               path_to_embed_model=os.path.join(MODELS_DIR, "EmoMulti_Embedding", "embedding_function.pt"),
               fine_tune=finetune,
               resume=resume,
               steps=40000,
               use_wandb=use_wandb,
               sent_embs=sent_embs,
               random_emb=True,
               emovdb=True)
    if use_wandb:
        wandb.finish()
