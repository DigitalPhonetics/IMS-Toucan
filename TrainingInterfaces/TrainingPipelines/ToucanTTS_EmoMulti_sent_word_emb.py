import time

import torch
import wandb
from torch.utils.data import ConcatDataset

from TrainingInterfaces.Text_to_Spectrogram.ToucanTTS.ToucanTTS import ToucanTTS
from TrainingInterfaces.Text_to_Spectrogram.ToucanTTS.toucantts_train_loop_arbiter import train_loop
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

    name = "ToucanTTS_12_EmoMulti_sent_word_emb_emoBERTcls_static_SE2_each"

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

    datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_EmoV_DB_Speaker(),
                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "emovdb_speaker"),
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
    
    try:
        transcript_dict_ljspeech = torch.load(os.path.join(PREPROCESSING_DIR, "ljspeech", "path_to_transcript_dict.pt"), map_location='cpu')
    except FileNotFoundError:
        transcript_dict_ljspeech = build_path_to_transcript_dict_ljspeech()

    datasets.append(prepare_fastspeech_corpus(transcript_dict=transcript_dict_ljspeech,
                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "ljspeech"),
                                          lang="en",
                                          save_imgs=False))
    
    train_set = ConcatDataset(datasets)

    if "_xvect" in name:
        if not os.path.exists(os.path.join(PREPROCESSING_DIR, "xvect_emomulti", "xvect.pt")):
            print("Extracting xvect from audio")
            os.makedirs(os.path.join(PREPROCESSING_DIR, "xvect_emomulti"), exist_ok=True)
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
    
    if "_ecapa" in name:
        if not os.path.exists(os.path.join(PREPROCESSING_DIR, "ecapa_emomulti", "ecapa.pt")):
            print("Extracting ecapa from audio")
            import torchaudio
            from speechbrain.pretrained import EncoderClassifier
            classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="./Models/Embedding/spkrec-ecapa-voxceleb", run_opts={"device": device})
            path_to_ecapa = {}
            for index in tqdm(range(len(train_set))):
                path = train_set[index][10]
                wave, sr = torchaudio.load(path)
                # mono
                wave = torch.mean(wave, dim=0, keepdim=True)
                # resampling
                wave = torchaudio.functional.resample(wave, orig_freq=sr, new_freq=16000)
                wave = wave.squeeze(0)
                embedding = classifier.encode_batch(wave).squeeze(0).squeeze(0)
                path_to_ecapa[path] = embedding
            torch.save(path_to_ecapa, os.path.join(PREPROCESSING_DIR, "ecapa_emomulti", "ecapa.pt"))
            del classifier
        else:
            print(f"Loading ecapa embeddings from {os.path.join(PREPROCESSING_DIR, 'ecapa_emomulti', 'ecapa.pt')}")
            path_to_ecapa = torch.load(os.path.join(PREPROCESSING_DIR, "ecapa_emomulti", "ecapa.pt"), map_location='cpu')
    else:
        path_to_ecapa = None
    if path_to_ecapa is not None:
        path_to_xvect = path_to_ecapa

    print(f'Loading sentence embeddings from {os.path.join(PREPROCESSING_DIR, "Yelp", f"emotion_prompts_large_sent_embs_emoBERTcls.pt")}')
    sent_embs = torch.load(os.path.join(PREPROCESSING_DIR, "Yelp", f"emotion_prompts_large_sent_embs_emoBERTcls.pt"), map_location='cpu')

    from Preprocessing.word_embeddings.BERTWordEmbeddingExtractor import BERTWordEmbeddingExtractor
    word_embedding_extractor = BERTWordEmbeddingExtractor()

    model = ToucanTTS(lang_embs=None, 
                      utt_embed_dim=512,
                      sent_embed_dim=768,
                      static_speaker_embed=True,
                      word_embed_dim=768)

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
               batch_size=16,
               eval_lang="en",
               path_to_checkpoint=resume_checkpoint,
               path_to_embed_model=None,
               fine_tune=finetune,
               resume=resume,
               use_wandb=use_wandb,
               sent_embs=sent_embs,
               random_emb=True,
               path_to_xvect=path_to_xvect,
               static_speaker_embed=True,
               word_embedding_extractor=word_embedding_extractor)
    if use_wandb:
        wandb.finish()