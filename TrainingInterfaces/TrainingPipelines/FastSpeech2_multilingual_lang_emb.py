import random

import torch
from torch.utils.data import ConcatDataset

from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2
#from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.fastspeech2_train_loop import train_loop
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.meta_train_loop import train_loop
from Utility.corpus_preparation import prepare_fastspeech_corpus
from Utility.path_to_transcript_dicts import *


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume):
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

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join("Models", "FastSpeech2_Meta_From_Labels_multi_lang_emb")
    os.makedirs(save_dir, exist_ok=True)


    austrian_datasets = list()
    german_datasets = list()
    spanish_datasets = list()
    dutch_datasets = list()
    french_datasets = list()
    portuguese_datasets = list()
    polish_datasets = list()
    italian_datasets = list()
    english_datasets = list()

    print("making polish dataset")
    polish_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_mls_polish(),
                                                     corpus_dir=os.path.join("Corpora", "mls_polish"),
                                                     lang="pl"))

    # polish_datasets.append(prepare_fastspeech_corpus(transcript_dict=dict(list(build_path_to_transcript_dict_mls_polish().items())[0:100]),
    #                                                  corpus_dir=os.path.join("Corpora", "mls_polish"),
    #                                                  lang="pl"))
    print("making austrian dataset")
    austrian_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_aridialect_input_phonemes(),
                                                       corpus_dir=os.path.join("Corpora", "austrian_from_labels_lang_emb"),
                                                       lang="de",
                                                       phone_input=True))
    print("making mls_german the other datasets")
    german_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_mls_german(),
                                                       corpus_dir=os.path.join("Corpora", "mls_german"),
                                                       lang="de"))
                                                       
    print("making mls_spanish the other datasets")
    spanish_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_mls_spanish(),
                                                      corpus_dir=os.path.join("Corpora", "mls_spanish"),
                                                      lang="es"))

    dutch_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_mls_dutch(),
                                                    corpus_dir=os.path.join("Corpora", "mls_dutch"),
                                                    lang="nl"))

    french_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_mls_french(),
                                                     corpus_dir=os.path.join("Corpora", "mls_french"),
                                                     lang="fr"))

    portuguese_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_mls_portuguese(),
                                                         corpus_dir=os.path.join("Corpora", "mls_porto"),
                                                         lang="pt"))



    italian_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_mls_italian(),
                                                      corpus_dir=os.path.join("Corpora", "mls_italian"),
                                                      lang="it"))

    english_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_mls_english(),
                                                      corpus_dir=os.path.join("Corpora", "mls_english"),
                                                      lang="en"))

    datasets = list()

    # datasets.append(ConcatDataset(random.sample(austrian_datasets,100)))
    # datasets.append(ConcatDataset(random.sample(german_datasets,100)))
    # datasets.append(ConcatDataset(random.sample(spanish_datasets,100)))
    # datasets.append(ConcatDataset(random.sample(dutch_datasets,100)))
    # datasets.append(ConcatDataset(random.sample(french_datasets,100)))
    # datasets.append(ConcatDataset(random.sample(portuguese_datasets,100)))
    # datasets.append(ConcatDataset(random.sample(polish_datasets,100)))
    # datasets.append(ConcatDataset(random.sample(italian_datasets,100)))
    # datasets.append(ConcatDataset(random.sample(english_datasets,100)))

    datasets.append(ConcatDataset(austrian_datasets))
    datasets.append(ConcatDataset(german_datasets))
    datasets.append(ConcatDataset(spanish_datasets))
    datasets.append(ConcatDataset(dutch_datasets))
    datasets.append(ConcatDataset(french_datasets))
    datasets.append(ConcatDataset(portuguese_datasets))
    datasets.append(ConcatDataset(polish_datasets))
    datasets.append(ConcatDataset(italian_datasets))
    datasets.append(ConcatDataset(english_datasets))
    
    #warmup steps reduced!!!!!!!!!!!!!!!!!!!!!
    #mini dataset used right now
    
    print("Training model")
    # train_loop(net=FastSpeech2(lang_embs=100),
    #            datasets=datasets,
    #            device=device,
    #            save_directory=save_dir,
    #            steps=500000,
    #            batch_size=1,
    #            lr=0.0001,
    #            steps_per_checkpoint=1000,
    #            warmup_steps=4000,
    #            path_to_checkpoint=None,
    #            resume=resume)

    train_loop(net=FastSpeech2(lang_embs=100),
               device=torch.device("cuda"),
               datasets=datasets,
               batch_size=6,
               save_directory=save_dir,
               steps=300000,
               steps_per_checkpoint=1000,
               lr=0.001,
               path_to_checkpoint=None,
               resume=resume)
