import time

import torch
import wandb
from torch.utils.data import ConcatDataset

from TrainingInterfaces.Text_to_Spectrogram.ToucanTTS.ToucanTTS import ToucanTTS
from TrainingInterfaces.Text_to_Spectrogram.ToucanTTS.toucantts_train_loop_arbiter import train_loop
from Utility.corpus_preparation import prepare_fastspeech_corpus
from Utility.path_to_transcript_dicts import *
from Utility.sent_emb_extraction import extract_sent_embs
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

    name = "ToucanTTS_Sent_LJSpeech"

    '''
    concat speaker embedding and sentence embedding
    input for encoder, pitch, energy, variance predictors and decoder
    '''

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join(MODELS_DIR, name)
    os.makedirs(save_dir, exist_ok=True)

    datasets = list()

    try:
        transcript_dict_ljspeech = torch.load(os.path.join(PREPROCESSING_DIR, "ljspeech", "path_to_transcript_dict.pt"), map_location='cpu')
    except FileNotFoundError:
        transcript_dict_ljspeech = build_path_to_transcript_dict_ljspeech()

    datasets.append(prepare_fastspeech_corpus(transcript_dict=transcript_dict_ljspeech,
                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "ljspeech"),
                                          lang="en",
                                          save_imgs=False))
    
    train_set = ConcatDataset(datasets)

    if not os.path.exists(os.path.join(PREPROCESSING_DIR, "ljspeech", "sent_embs_emoBERTcls.pt")):
        from Preprocessing.sentence_embeddings.EmotionRoBERTaSentenceEmbeddingExtractor import EmotionRoBERTaSentenceEmbeddingExtractor as SentenceEmbeddingExtractor
        sentence_embedding_extractor = SentenceEmbeddingExtractor(pooling="cls")
        sent_embs = extract_sent_embs(train_set=train_set, sent_emb_extractor=sentence_embedding_extractor)
        atf = ArticulatoryCombinedTextFrontend(language="en")
        example_sentence = atf.get_example_sentence(lang="en")
        sent_embs[example_sentence] = sentence_embedding_extractor.encode(sentences=[example_sentence]).squeeze()
        torch.save(sent_embs, os.path.join(PREPROCESSING_DIR, "ljspeech", "sent_embs_emoBERTcls.pt"))
        print(f'Saved sentence embeddings in {os.path.join(PREPROCESSING_DIR, "ljspeech", "sent_embs_emoBERTcls.pt")}')
        del sentence_embedding_extractor
    else:
        print(f'Loading sentence embeddings from {os.path.join(PREPROCESSING_DIR, "ljspeech", "sent_embs_emoBERTcls.pt")}.')
        sent_embs = torch.load(os.path.join(PREPROCESSING_DIR, "ljspeech", "sent_embs_emoBERTcls.pt"), map_location='cpu')

    return

    model = ToucanTTS(lang_embs=None, 
                      utt_embed_dim=512,
                      sent_embed_dim=768,
                      static_speaker_embed=True)

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
               batch_size=32,
               eval_lang="en",
               path_to_checkpoint=resume_checkpoint,
               path_to_embed_model=None,
               fine_tune=finetune,
               resume=resume,
               use_wandb=use_wandb,
               sent_embs=sent_embs,
               path_to_xvect=None,
               static_speaker_embed=True)
    if use_wandb:
        wandb.finish()