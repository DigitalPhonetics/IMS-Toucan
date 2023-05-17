import time

import torch
import wandb

from TrainingInterfaces.Text_to_Embedding.SentenceEmbeddingAdaptor import SentenceEmbeddingAdaptor
from TrainingInterfaces.Text_to_Embedding.sent_emb_adaptor_train_loop import train_loop
from Utility.corpus_preparation import prepare_fastspeech_corpus
from Utility.sent_emb_extraction import extract_sent_embs
from Utility.path_to_transcript_dicts import *
from Utility.storage_config import MODELS_DIR
from Utility.storage_config import PREPROCESSING_DIR
from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend

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

    name = "SentEmbAdaptor_01_EmoVDBSam_emoBERTcls_yelp"

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join(MODELS_DIR, name)
    os.makedirs(save_dir, exist_ok=True)

    train_set = prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_emovdb_sam(),
                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "emovdb_sam"),
                                          lang="en",
                                          save_imgs=False)
    
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
        if "yelp" in name:
            embed_type = "emoBERTcls_yelp"
        else:
            embed_type = "emoBERTcls"
        sent_embed_dim = 768

    if not os.path.exists(os.path.join(PREPROCESSING_DIR, "emovdb_sam", f"sent_emb_cache_{embed_type}.pt")):
        if embed_type == "lealla":
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            from Preprocessing.sentence_embeddings.LEALLASentenceEmbeddingExtractor import LEALLASentenceEmbeddingExtractor as SentenceEmbeddingExtractor
            sentence_embedding_extractor = SentenceEmbeddingExtractor()
        if embed_type == "laser":
            from Preprocessing.sentence_embeddings.LASERSentenceEmbeddingExtractor import LASERSentenceEmbeddingExtractor as SentenceEmbeddingExtractor
            sentence_embedding_extractor = SentenceEmbeddingExtractor()
        if embed_type == "para":
            from Preprocessing.sentence_embeddings.STSentenceEmbeddingExtractor import STSentenceEmbeddingExtractor as SentenceEmbeddingExtractor
            sentence_embedding_extractor = SentenceEmbeddingExtractor(model="para")
        if embed_type == "mpnet":
            from Preprocessing.sentence_embeddings.STSentenceEmbeddingExtractor import STSentenceEmbeddingExtractor as SentenceEmbeddingExtractor
            sentence_embedding_extractor = SentenceEmbeddingExtractor(model="mpnet")
        if embed_type == "bertcls":
            from Preprocessing.sentence_embeddings.BERTSentenceEmbeddingExtractor import BERTSentenceEmbeddingExtractor as SentenceEmbeddingExtractor
            sentence_embedding_extractor = SentenceEmbeddingExtractor(pooling="cls")
        if embed_type == "bertlm":
            from Preprocessing.sentence_embeddings.BERTSentenceEmbeddingExtractor import BERTSentenceEmbeddingExtractor as SentenceEmbeddingExtractor
            sentence_embedding_extractor = SentenceEmbeddingExtractor(pooling="last_mean")
        if embed_type == "emoBERTcls":
            from Preprocessing.sentence_embeddings.EmotionRoBERTaSentenceEmbeddingExtractor import EmotionRoBERTaSentenceEmbeddingExtractor as SentenceEmbeddingExtractor
            sentence_embedding_extractor = SentenceEmbeddingExtractor(pooling="cls")

        sent_embs = extract_sent_embs(train_set=train_set, sent_emb_extractor=sentence_embedding_extractor, emovdb=True)
        sent_embs["example_sentence"] = sentence_embedding_extractor.encode(sentences=["This is a test sentence."]).squeeze()
        torch.save(sent_embs, os.path.join(PREPROCESSING_DIR, "emovdb_sam", f"sent_emb_cache_{embed_type}.pt"))
        print(f'Saved sentence embeddings in {os.path.join(PREPROCESSING_DIR, "emovdb_sam", f"sent_emb_cache_{embed_type}.pt")}')
        if embed_type == "lealla":
            print("Please restart and use saved sentence embeddings because tensorflow won't release GPU memory for training.")
            return
        else:
            del sentence_embedding_extractor
    else:
        print(f'Loading sentence embeddings from {os.path.join(PREPROCESSING_DIR, "emovdb_sam", f"sent_emb_cache_{embed_type}.pt")}.')
        sent_embs = torch.load(os.path.join(PREPROCESSING_DIR, "emovdb_sam", f"sent_emb_cache_{embed_type}.pt"), map_location='cpu')
    
    if sent_embs is None:
        raise TypeError("Sentence embeddings are None.")
    
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
               path_to_embed_model=os.path.join(MODELS_DIR, "EmoVDBSam_Embedding", "embedding_function.pt"),
               fine_tune=finetune,
               resume=resume,
               steps=200000,
               use_wandb=use_wandb,
               sent_embs=sent_embs,
               random_emb="yelp" in name,
               emovdb=True)
    if use_wandb:
        wandb.finish()
