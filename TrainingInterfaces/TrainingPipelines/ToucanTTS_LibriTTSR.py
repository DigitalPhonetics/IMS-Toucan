import time

import torch
import wandb
from torch.utils.data import ConcatDataset
from tqdm import tqdm

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
        device = torch.device("cuda")

    torch.manual_seed(131714)
    random.seed(131714)
    torch.random.manual_seed(131714)

    print("Preparing")

    name = "ToucanTTS_Baseline_LibriTTSR"

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join(MODELS_DIR, name)
    os.makedirs(save_dir, exist_ok=True)

    datasets = list()

    try:
        transcript_dict_librittsr = torch.load(os.path.join(PREPROCESSING_DIR, "librittsr", "path_to_transcript_dict.pt"), map_location='cpu')
    except FileNotFoundError:
        transcript_dict_librittsr = build_path_to_transcript_dict_libritts_all_clean()

    datasets.append(prepare_fastspeech_corpus(transcript_dict=transcript_dict_librittsr,
                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "librittsr"),
                                          lang="en",
                                          save_imgs=False))
    
    train_set = ConcatDataset(datasets)

    if "_xvect" in name:
        if not os.path.exists(os.path.join(PREPROCESSING_DIR, "xvect_emomulti", "xvect.pt")):
            print("Extracting xvect from audio")
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

    model = ToucanTTS(lang_embs=None, 
                      utt_embed_dim=512,
                      static_speaker_embed=True)
    if use_wandb:
        wandb.init(
            name=f"{name}_{time.strftime('%Y%m%d-%H%M%S')}" if wandb_resume_id is None else None,
            id=wandb_resume_id,  # this is None if not specified in the command line arguments.
            resume="must" if wandb_resume_id is not None else None)
    print("Training model")
    train_loop(net=model,
               datasets=train_set,
               device=device,
               save_directory=save_dir,
               batch_size=16,
               eval_lang="en",
               path_to_checkpoint=resume_checkpoint,
               path_to_embed_model=None,
               fine_tune=finetune,
               resume=resume,
               use_wandb=use_wandb,
               path_to_xvect=None,
               static_speaker_embed=True)
    if use_wandb:
        wandb.finish()