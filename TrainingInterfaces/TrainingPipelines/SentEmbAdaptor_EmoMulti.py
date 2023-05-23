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

    name = "SentEmbAdaptor_01_EmoMulti_emoBERTcls_xvect"

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

    if "_xvect" in name:
        if not os.path.exists(os.path.join(PREPROCESSING_DIR, "xvect_emomulti", "xvect.pt")):
            print("Extracting xvect from audio")
            # TODO run on GPU
            import torchaudio
            from speechbrain.pretrained import EncoderClassifier
            classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="./Models/Embedding/spkrec-xvect-voxceleb", run_opts={"device": device})
            path_to_xvect = {}
            for index in tqdm(range(len(train_set))):
                path = train_set[index][10]
                wave, sr = torchaudio.load(path)
                # mono
                wave = torch.mean(wave, dim=0, keepdim=True)
                # resampling
                wave = torchaudio.functional.resample(wave, orig_freq=sr, new_freq=16000)
                wave = wave.squeeze(0)
                embedding = classifier.encode_batch(wave).squeeze(0).squeeze(0)
                path_to_xvect[path] = embedding
            torch.save(path_to_xvect, os.path.join(PREPROCESSING_DIR, "xvect_emomulti", "xvect.pt"))
            del classifier
        else:
            print(f"Loading xvect embeddings from {os.path.join(PREPROCESSING_DIR, 'xvect_emomulti', 'xvect.pt')}")
            path_to_xvect = torch.load(os.path.join(PREPROCESSING_DIR, "xvect_emomulti", "xvect.pt"), map_location='cpu')
    else:
        path_to_xvect = None

    print(f'Loading sentence embeddings from {os.path.join(PREPROCESSING_DIR, "Yelp", f"emotion_prompts_large_sent_embs_{embed_type}.pt")}')
    sent_embs = torch.load(os.path.join(PREPROCESSING_DIR, "Yelp", f"emotion_prompts_large_sent_embs_{embed_type}.pt"), map_location='cpu')
    
    model = SentenceEmbeddingAdaptor(sent_embed_dim=sent_embed_dim, 
                                     utt_embed_dim=64, 
                                     speaker_embed_dim=512 if "_xvect" in name else None)

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
               emovdb=True,
               path_to_xvect=path_to_xvect)
    if use_wandb:
        wandb.finish()
