import time

import torch
import wandb
from torch.utils.data import ConcatDataset

from TrainingInterfaces.Text_to_Spectrogram.ToucanTTS.ToucanTTS import ToucanTTS
from TrainingInterfaces.Text_to_Spectrogram.ToucanTTS.toucantts_train_loop_arbiter import train_loop
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

    name = "ToucanTTS_06_CremaD_sent_emb_a11_emoBERTcls_static"
    """
    a01: integrate before encoder
    a02: integrate before encoder and decoder
    a03: integrate before encoder and decoder and postnet
    a04: integrate before each encoder layer
    a05: integrate before each encoder and decoder layer
    a06: integrate before each encoder and decoder layer and postnet
    a07: concatenate with style embedding and apply projection
    a08: concatenate with style embedding
    a09: a06 + a07
    a10: replace style embedding with sentence embedding (no style embedding, no language embedding, single speaker single language case)
    a11: a01 + a07
    a12: integrate before encoder and use sentence embedding instead of style embedding (can be constrained with loss)
    a13: use sentence embedding instead of style embedding (can be constrained with loss or adaptor)
    loss: additionally use sentence style loss
    """

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join(MODELS_DIR, name)
    os.makedirs(save_dir, exist_ok=True)

    datasets = list()

    datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_CREMA_D(),
                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "cremad"),
                                          lang="en",
                                          save_imgs=False))
    
    train_set = ConcatDataset(datasets)

    if "_xvect" in name:
        print(f"Loading xvect embeddings from {os.path.join(PREPROCESSING_DIR, 'xvect_emomulti', 'xvect.pt')}")
        path_to_xvect = torch.load(os.path.join(PREPROCESSING_DIR, "xvect_emomulti", "xvect.pt"), map_location='cpu')
    else:
        path_to_xvect = None

    if "_ecapa" in name:
        print(f"Loading ecapa embeddings from {os.path.join(PREPROCESSING_DIR, 'ecapa_emomulti', 'ecapa.pt')}")
        path_to_ecapa = torch.load(os.path.join(PREPROCESSING_DIR, "ecapa_emomulti", "ecapa.pt"), map_location='cpu')
    else:
        path_to_ecapa = None
    if path_to_ecapa is not None:
        path_to_xvect = path_to_ecapa

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

    sent_embed_encoder=False
    sent_embed_decoder=False
    sent_embed_each=False
    sent_embed_postnet=False
    concat_sent_style=False
    use_concat_projection=False
    replace_utt_sent_emb = False
    style_sent = False

    lang_embs=None
    if "_xvect" in name and "_adapted" not in name:
        utt_embed_dim = 512
    elif "_ecapa" in name and "_adapted" not in name:
        utt_embed_dim = 192
    else:
        utt_embed_dim = 64

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
    if "a09" in name:
        sent_embed_encoder=True
        sent_embed_decoder=True
        sent_embed_each=True
        sent_embed_postnet=True
        concat_sent_style=True
        use_concat_projection=True
    if "a10" in name:
        lang_embs = None
        utt_embed_dim = 192
        sent_embed_dim = None
        replace_utt_sent_emb = True
    if "a11" in name:
        sent_embed_encoder=True
        concat_sent_style=True
        use_concat_projection=True
    if "a12" in name:
        sent_embed_encoder=True
        style_sent=True
        if "noadapt" in name and "adapted" not in name:
            utt_embed_dim = 768
    if "a13" in name:
        style_sent=True
        utt_embed_dim = sent_embed_dim
        if "noadapt" in name and "adapted" not in name:
            utt_embed_dim = 768


    model = ToucanTTS(lang_embs=lang_embs, 
                    utt_embed_dim=utt_embed_dim,
                    sent_embed_dim=64 if "adapted" in name else sent_embed_dim,
                    sent_embed_adaptation="noadapt" not in name,
                    sent_embed_encoder=sent_embed_encoder,
                    sent_embed_decoder=sent_embed_decoder,
                    sent_embed_each=sent_embed_each,
                    sent_embed_postnet=sent_embed_postnet,
                    concat_sent_style=concat_sent_style,
                    use_concat_projection=use_concat_projection,
                    use_sent_style_loss="loss" in name,
                    pre_embed="_pre" in name,
                    style_sent=style_sent,
                    static_speaker_embed="_static" in name)

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
               path_to_embed_model=os.path.join(MODELS_DIR, "EmoMulti_Embedding", "embedding_function.pt"),
               fine_tune=finetune,
               resume=resume,
               use_wandb=use_wandb,
               sent_embs=sent_embs,
               random_emb=True,
               emovdb=True,
               replace_utt_sent_emb=replace_utt_sent_emb,
               use_adapted_embs="adapted" in name,
               path_to_xvect=path_to_xvect,
               static_speaker_embed="_static" in name)
    if use_wandb:
        wandb.finish()