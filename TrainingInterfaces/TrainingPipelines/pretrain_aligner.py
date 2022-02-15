import random

import torch
from torch.utils.data import ConcatDataset

from TrainingInterfaces.Text_to_Spectrogram.AutoAligner.autoaligner_train_loop import train_loop as train_aligner
from Utility.corpus_preparation import prepare_aligner_corpus
from Utility.path_to_transcript_dicts import *


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume):
    if gpu_id == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")

    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
        device = torch.device("cuda")

    torch.manual_seed(131714)
    random.seed(131714)
    torch.random.manual_seed(131714)

    print("Preparing")

    datasets = list()

    datasets.append(prepare_aligner_corpus(transcript_dict=dict(random.sample(build_path_to_transcript_dict_mls_portuguese().items(), 20000)),
                                           # take only 20k samples from this, since the corpus is way too big,
                                           corpus_dir=os.path.join("Corpora", "mls_porto"),
                                           lang="pt",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=dict(random.sample(build_path_to_transcript_dict_mls_polish().items(), 20000)),
                                           # take only 20k samples from this, since the corpus is way too big,
                                           corpus_dir=os.path.join("Corpora", "mls_polish"),
                                           lang="pl",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=dict(random.sample(build_path_to_transcript_dict_mls_spanish().items(), 12000)),
                                           # take only 12k samples from this, since the corpus is way too big,
                                           corpus_dir=os.path.join("Corpora", "mls_spanish"),
                                           lang="es",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=dict(random.sample(build_path_to_transcript_dict_mls_french().items(), 12000)),
                                           # take only 12k samples from this, since the corpus is way too big
                                           corpus_dir=os.path.join("Corpora", "mls_french"),
                                           lang="fr",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=dict(random.sample(build_path_to_transcript_dict_mls_italian().items(), 20000)),
                                           # take only 20k samples from this, since the corpus is way too big,
                                           corpus_dir=os.path.join("Corpora", "mls_italian"),
                                           lang="it",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=dict(random.sample(build_path_to_transcript_dict_mls_dutch().items(), 12000)),
                                           # take only 12k samples from this, since the corpus is way too big
                                           corpus_dir=os.path.join("Corpora", "mls_dutch"),
                                           lang="nl",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_nancy(),
                                           corpus_dir=os.path.join("Corpora", "Nancy"),
                                           lang="en",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_karlsson(),
                                           corpus_dir=os.path.join("Corpora", "Karlsson"),
                                           lang="de",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10el(),
                                           corpus_dir=os.path.join("Corpora", "meta_Greek"),
                                           lang="el",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10es(),
                                           corpus_dir=os.path.join("Corpora", "meta_Spanish"),
                                           lang="es",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10fi(),
                                           corpus_dir=os.path.join("Corpora", "meta_Finnish"),
                                           lang="fi",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10ru(),
                                           corpus_dir=os.path.join("Corpora", "meta_Russian"),
                                           lang="ru",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10hu(),
                                           corpus_dir=os.path.join("Corpora", "meta_Hungarian"),
                                           lang="hu",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10nl(),
                                           corpus_dir=os.path.join("Corpora", "meta_Dutch"),
                                           lang="nl",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10fr(),
                                           corpus_dir=os.path.join("Corpora", "meta_French"),
                                           lang="fr",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_ljspeech(),
                                           corpus_dir=os.path.join("Corpora", "LJSpeech"),
                                           lang="en",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_libritts(),
                                           corpus_dir=os.path.join("Corpora", "libri"),
                                           lang="en",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=dict(random.sample(build_path_to_transcript_dict_vctk().items(), 20000)),
                                           # take only 20k samples from this, since the corpus is way too big,
                                           corpus_dir=os.path.join("Corpora", "vctk"),
                                           lang="en",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=dict(random.sample(build_path_to_transcript_dict_nvidia_hifitts().items(), 20000)),
                                           # take only 20k samples from this, since the corpus is way too big,
                                           corpus_dir=os.path.join("Corpora", "hifi"),
                                           lang="en",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_spanish_blizzard_train(),
                                           corpus_dir=os.path.join("Corpora", "spanish_blizzard"),
                                           lang="es",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_eva(),
                                           corpus_dir=os.path.join("Corpora", "Eva"),
                                           lang="de",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=dict(random.sample(build_path_to_transcript_dict_hokus().items(), 10000)),
                                           # take only 10k samples from this
                                           corpus_dir=os.path.join("Corpora", "Hokus"),
                                           lang="de",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=dict(random.sample(build_path_to_transcript_dict_bernd().items(), 12000)),
                                           # take only 12k samples from this, since the corpus is way too big,
                                           corpus_dir=os.path.join("Corpora", "Bernd"),
                                           lang="de",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_hui_others(),
                                           corpus_dir=os.path.join("Corpora", "hui_others"),
                                           lang="de",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=dict(random.sample(build_path_to_transcript_dict_thorsten().items(), 12000)),
                                           # take only 12k samples from this, since the corpus is not that high quality,
                                           corpus_dir=os.path.join("Corpora", "Thorsten"),
                                           lang="de",
                                           device=device))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fluxsing(),
                                           corpus_dir=os.path.join("Corpora", "flux_sing"),
                                           lang="en",
                                           device=device))

    train_set = ConcatDataset(datasets)
    save_dir = os.path.join("Models", "Aligner")
    os.makedirs(save_dir, exist_ok=True)
    save_dir_aligner = save_dir + "/aligner"
    os.makedirs(save_dir_aligner, exist_ok=True)

    train_aligner(train_dataset=train_set,
                  device=device,
                  save_directory=save_dir,
                  steps=500000,
                  batch_size=32,
                  path_to_checkpoint=resume_checkpoint,
                  fine_tune=finetune,
                  debug_img_path=save_dir_aligner,
                  resume=resume)
