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

    # datasets.append(prepare_aligner_corpus(transcript_dict=dict(random.sample(build_path_to_transcript_dict_aridialect_input_phonemes().items(), 20000)),
    #                                        # take only 20k samples from this, since the corpus is way too big,
    #                                        corpus_dir=os.path.join("Corpora", "austrian_from_labels_lang_emb"),
    #                                        lang="de",
    #                                        device=device,
    #                                        phone_input=True))
    print("make german aligner")
    # datasets.append(prepare_aligner_corpus(transcript_dict=dict(random.sample(build_path_to_transcript_dict_mls_german().items(), 20000)),
    #                                        # take only 20k samples from this, since the corpus is way too big,
    #                                        corpus_dir=os.path.join("Corpora", "mls_german"),
    #                                        lang="de",
    #                                        device=device))
    # print("make portuguese aligner")
    # datasets.append(prepare_aligner_corpus(transcript_dict=dict(random.sample(build_path_to_transcript_dict_mls_portuguese().items(), 20000)),
    #                                        corpus_dir=os.path.join("Corpora", "mls_porto"),
    #                                        lang="pt",
    #                                        device=device))
    # datasets.append(prepare_aligner_corpus(transcript_dict=dict(random.sample(build_path_to_transcript_dict_mls_polish().items(), 20000)),
    #                                        corpus_dir=os.path.join("Corpora", "mls_polish"),
    #                                        lang="pl",
    #                                        device=device))
    print("make spanish aligner")
    datasets.append(prepare_aligner_corpus(transcript_dict=dict(random.sample(build_path_to_transcript_dict_mls_spanish().items(), 12000)),
                                           corpus_dir=os.path.join("Corpora", "mls_spanish"),
                                           lang="es",
                                           device=device))
    print("make french aligner")
    datasets.append(prepare_aligner_corpus(transcript_dict=dict(random.sample(build_path_to_transcript_dict_mls_french().items(), 12000)),
                                           corpus_dir=os.path.join("Corpora", "mls_french"),
                                           lang="fr",
                                           device=device))
    print("make italian aligner")
    datasets.append(prepare_aligner_corpus(transcript_dict=dict(random.sample(build_path_to_transcript_dict_mls_italian().items(), 20000)),
                                           corpus_dir=os.path.join("Corpora", "mls_italian"),
                                           lang="it",
                                           device=device))
    print("make dutch aligner")
    datasets.append(prepare_aligner_corpus(transcript_dict=dict(random.sample(build_path_to_transcript_dict_mls_dutch().items(), 12000)),
                                           corpus_dir=os.path.join("Corpora", "mls_dutch"),
                                           lang="nl",
                                           device=device))
    print("make english aligner")
    datasets.append(prepare_aligner_corpus(transcript_dict=dict(random.sample(build_path_to_transcript_dict_mls_english().items(), 20000)),
                                           corpus_dir=os.path.join("Corpora", "mls_english"),
                                           lang="en",
                                           device=device))
    print("finished making alignment pretraining!")
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
