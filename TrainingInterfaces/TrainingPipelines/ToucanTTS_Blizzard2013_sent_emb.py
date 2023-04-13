import time

import torch
import wandb
import tensorflow as tf

from TrainingInterfaces.Text_to_Spectrogram.ToucanTTS.ToucanTTS import ToucanTTS
from TrainingInterfaces.Text_to_Spectrogram.ToucanTTS.toucantts_train_loop_arbiter import train_loop
from Utility.corpus_preparation import prepare_fastspeech_corpus
from Utility.sent_emb_extraction import extract_sent_embs
from Utility.path_to_transcript_dicts import *
from Utility.storage_config import MODELS_DIR
from Utility.storage_config import PREPROCESSING_DIR
from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend

import sys


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume, use_wandb, wandb_resume_id):
    if gpu_id == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")

    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
        device = torch.device(f"cuda")
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

    torch.manual_seed(131714)
    random.seed(131714)
    torch.random.manual_seed(131714)

    print("Preparing")

    name = "01_ToucanTTS_Blizzard2013_sent_emb_a04"
    """
    a01: integrate before encoder
    a02: integrate before encoder and decoder
    a03: integrate before encoder and decoder and postnet
    a04: integrate before each encoder layer
    a05: integrate before each encoder and decoder layer
    a06: integrate before each encoder and decoder layer and postnet
    a07: concatenate with style embedding and apply projection
    a08: concatenate with style embedding
    loss: additionally use sentence style loss
    """

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join(MODELS_DIR, name)
    os.makedirs(save_dir, exist_ok=True)

    train_set = prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_blizzard_2013(),
                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "blizzard2013"),
                                          lang="en",
                                          save_imgs=False)
    if not os.path.exists(os.path.join(PREPROCESSING_DIR, "blizzard2013", "sent_emb_cache.pt")):
        from Preprocessing.sentence_embeddings.LEALLASentenceEmbeddingExtractor import LEALLASentenceEmbeddingExtractor as SentenceEmbeddingExtractor
        sentence_embedding_extractor = SentenceEmbeddingExtractor()
        sent_embs = extract_sent_embs(train_set=train_set, sent_emb_extractor=sentence_embedding_extractor)
        atf = ArticulatoryCombinedTextFrontend(language="en")
        example_sentence = atf.get_example_sentence(lang="en")
        sent_embs[example_sentence] = sentence_embedding_extractor.encode(sentences=[example_sentence]).squeeze()
        torch.save(sent_embs, os.path.join(PREPROCESSING_DIR, "blizzard2013", "sent_emb_cache.pt"))
        print(f'Saved sentence embeddings in {os.path.join(PREPROCESSING_DIR, "blizzard2013", "sent_emb_cache.pt")}')
        return
    else:
        print(f'Loading sentence embeddings from {os.path.join(PREPROCESSING_DIR, "blizzard2013", "sent_emb_cache.pt")}.')
        sent_embs = torch.load(os.path.join(PREPROCESSING_DIR, "blizzard2013", "sent_emb_cache.pt"), map_location='cpu')
    
    if sent_embs is None:
        raise TypeError("Sentence embeddings are None.")

    sent_embed_adaptation=True
    sent_embed_encoder=False
    sent_embed_decoder=False
    sent_embed_each=False
    sent_embed_postnet=False
    concat_sent_style=False
    use_concat_projection=False

    if "a01" in name:
        sent_embed_encoder=True
    if "a02" in name:
        sent_embed_encoder=True
        sent_embed_decoder=True
    if "a03" in name:
        sent_embed_encoder=True
        sent_embed_decoder=True
        sent_embed_postnet=True
    if "a04" in name:
        sent_embed_encoder=True
        sent_embed_each=True
    if "a05" in name:
        sent_embed_encoder=True
        sent_embed_decoder=True
        sent_embed_each=True
    if "a06" in name:
        sent_embed_encoder=True
        sent_embed_decoder=True
        sent_embed_each=True
        sent_embed_postnet=True
    if "a07" in name:
        concat_sent_style=True
        use_concat_projection=True
    if "a08" in name:
        concat_sent_style=True

    lang_embs=8000
    utt_embed_dim=64
    sent_embed_dim=192

    model = ToucanTTS(lang_embs=lang_embs, 
                    utt_embed_dim=utt_embed_dim,
                    sent_embed_dim=sent_embed_dim,
                    sent_embed_adaptation=sent_embed_adaptation,
                    sent_embed_encoder=sent_embed_encoder,
                    sent_embed_decoder=sent_embed_decoder,
                    sent_embed_each=sent_embed_each,
                    sent_embed_postnet=sent_embed_postnet,
                    concat_sent_style=concat_sent_style,
                    use_concat_projection=use_concat_projection,
                    use_sent_style_loss="loss" in name)

    if use_wandb:
        wandb.init(
            name=f"{name}_{time.strftime('%Y%m%d-%H%M%S')}" if wandb_resume_id is None else None,
            id=wandb_resume_id,  # this is None if not specified in the command line arguments.
            resume="must" if wandb_resume_id is not None else None)
    print("Training model")
    train_loop(net=model,
               datasets=[train_set],
               device=device,
               save_directory=save_dir,
               batch_size=4,
               eval_lang="en",
               path_to_checkpoint=resume_checkpoint,
               path_to_embed_model=os.path.join(MODELS_DIR, "Embedding", "embedding_function.pt"),
               fine_tune=finetune,
               resume=resume,
               use_wandb=use_wandb,
               sent_embs=sent_embs)
    if use_wandb:
        wandb.finish()