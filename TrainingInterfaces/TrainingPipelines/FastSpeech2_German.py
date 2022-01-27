import random

import torch
from torch.utils.data import ConcatDataset

from TrainingInterfaces.Text_to_Spectrogram.AutoAligner.AlignerDataset import AlignerDataset
from TrainingInterfaces.Text_to_Spectrogram.AutoAligner.autoaligner_train_loop import train_loop as train_aligner
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeechDataset import FastSpeechDataset
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.fastspeech2_train_loop import train_loop
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

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join("Models", "FastSpeech2_German")
    os.makedirs(save_dir, exist_ok=True)

    datasets = list()
    datasets.append(prepare_corpus(transcript_dict=build_path_to_transcript_dict_karlsson(),
                                   corpus_dir=os.path.join("Corpora", "Karlsson"),
                                   lang="de"))

    datasets.append(prepare_corpus(transcript_dict=build_path_to_transcript_dict_eva(),
                                   corpus_dir=os.path.join("Corpora", "Eva"),
                                   lang="de"))

    datasets.append(prepare_corpus(transcript_dict=build_path_to_transcript_dict_hokus(),
                                   corpus_dir=os.path.join("Corpora", "Hokus"),
                                   lang="de"))

    datasets.append(prepare_corpus(transcript_dict=build_path_to_transcript_dict_bernd(),
                                   corpus_dir=os.path.join("Corpora", "Bernd"),
                                   lang="de"))

    datasets.append(prepare_corpus(transcript_dict=build_path_to_transcript_dict_friedrich(),
                                   corpus_dir=os.path.join("Corpora", "Friedrich"),
                                   lang="de"))

    datasets.append(prepare_corpus(transcript_dict=build_path_to_transcript_dict_hui_others(),
                                   corpus_dir=os.path.join("Corpora", "hui_others"),
                                   lang="de"))

    datasets.append(prepare_corpus(transcript_dict=build_path_to_transcript_dict_thorsten(),
                                   corpus_dir=os.path.join("Corpora", "Thorsten"),
                                   lang="de"))

    train_set = ConcatDataset(datasets)

    model = FastSpeech2(lang_embs=100)
    # even though it's monolingual, we initialize it with language embeddings, so that we can fine-tune from the meta-checkpoint.

    print("Training model")
    train_loop(net=model,
               train_dataset=train_set,
               device=device,
               save_directory=save_dir,
               steps=500000,
               batch_size=32,
               lang="de",
               lr=0.0001,
               warmup_steps=14000,
               path_to_checkpoint=resume_checkpoint,
               fine_tune=finetune,
               resume=resume)


def prepare_corpus(transcript_dict, corpus_dir, lang, ctc_selection=True):
    """
    create an aligner dataset,
    fine-tune an aligner,
    create a fastspeech dataset,
    return it.

    Skips parts that have been done before.
    """
    aligner_dir = os.path.join(corpus_dir, "aligner")
    if not os.path.exists(os.path.join(aligner_dir, "aligner.pt")):
        train_aligner(train_dataset=AlignerDataset(transcript_dict, cache_dir=corpus_dir, lang=lang),
                      device=torch.device("cuda"),
                      save_directory=aligner_dir,
                      steps=(len(transcript_dict.keys()) / 32) * 2,  # 3 epochs worth of finetuning
                      batch_size=32,
                      path_to_checkpoint="Models/Aligner/aligner.pt",
                      fine_tune=True,
                      debug_img_path=aligner_dir,
                      resume=False)
    return FastSpeechDataset(transcript_dict,
                             acoustic_checkpoint_path=os.path.join(aligner_dir, "aligner.pt"),
                             cache_dir=corpus_dir,
                             device=torch.device("cuda"),
                             lang=lang,
                             ctc_selection=ctc_selection)
