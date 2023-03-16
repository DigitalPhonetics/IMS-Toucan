import random

import torch
from torch.utils.data import ConcatDataset

from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.fastspeech2_train_loop import train_loop
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
        save_dir = os.path.join("Models", "FastSpeech2_Austrian_From_Labels")
    os.makedirs(save_dir, exist_ok=True)
    
    datasets = list()
    # to extract phonemes from label files, I build a function, that saves the phonemes to a specified folder.
    # set write_to_file=True in /Preprocessing/TextFrontend "def string_to_tensor(self, text, view=True, device="cpu", handle_missing=True, input_phonemes=False, path_to_wavfile="", write_to_file=False):"
    # 1. run function with build_path_to_transcript_dict_aridialect_and_write_phonemes_to_file_train()
    # 2. delete corpora or rename it.
    # 3. run function with build_path_to_transcript_dict_aridialect_and_write_phonemes_to_file_test()  
    datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_aridialect_and_write_phonemes_to_file_train(),
                                              corpus_dir=os.path.join("Corpora", "austrian_from_labels"),
                                              lang="at-lab",
                                              phone_input=False))
                                              
'''
    datasets = list()
    datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_aridialect_input_phonemes(),
                                              corpus_dir=os.path.join("Corpora", "austrian_from_labels"),
                                              lang="de",
                                              phone_input=True))
    train_set = ConcatDataset(datasets)

    
    model = FastSpeech2(lang_embs=100)
    # because we want to finetune it, we treat it as multilingual, even though we are only interested in German here

    print("Training model")
    train_loop(net=model,
               train_dataset=train_set,
               device=device,
               save_directory=save_dir,
               steps=500000,
               batch_size=8,
               lang="de",
               lr=0.001,
               epochs_per_save=1,
               warmup_steps=4000,
               path_to_checkpoint="Models/FastSpeech2_Meta/best.pt",
               fine_tune=True,
               resume=resume)
'''

