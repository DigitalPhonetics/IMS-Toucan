import random

import torch
from tqdm import tqdm

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
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
        device = torch.device("cuda")

    torch.manual_seed(131714)
    random.seed(131714)
    torch.random.manual_seed(131714)

    print("Preparing")

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join("Models", "FastSpeech2_LibriTTS_asr_phn")
    os.makedirs(save_dir, exist_ok=True)

    train_set = prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_libritts_asr_phn(),
                                          corpus_dir=os.path.join("Corpora", "libri_asr_phn"),
                                          lang="en",
                                          phone_input=True,
                                          ctc_selection=False)

    model = FastSpeech2()

    find_faulty_samples(net=model,
                        datasets=train_set,
                        device=torch.device("cuda"),
                        path_to_checkpoint="Models/FastSpeech2_LibriTTS/best.pt")

    import sys
    sys.exit()

    print("Training model")
    train_loop(net=model,
               train_dataset=train_set,
               device=device,
               save_directory=save_dir,
               steps=500000,
               batch_size=32,
               lang="en",
               lr=0.0001,
               warmup_steps=4000,
               path_to_checkpoint="Models/FastSpeech2_LibriTTS/best.pt",
               fine_tune=True,
               resume=resume)


@torch.inference_mode()
def find_faulty_samples(net,
                        datasets,
                        device,
                        path_to_checkpoint):
    net = net.to(device)
    torch.multiprocessing.set_sharing_strategy('file_system')
    check_dict = torch.load(os.path.join(path_to_checkpoint), map_location=device)
    net.load_state_dict(check_dict["model"])
    losses = list()
    index_pairs = list()
    for datapoint_index in tqdm(range(len(datasets))):
        loss = net(text_tensors=datasets[datapoint_index][0].unsqueeze(0).to(device),
                   text_lengths=datasets[datapoint_index][1].to(device),
                   gold_speech=datasets[datapoint_index][2].unsqueeze(0).to(device),
                   speech_lengths=datasets[datapoint_index][3].to(device),
                   gold_durations=datasets[datapoint_index][4].unsqueeze(0).to(device),
                   gold_pitch=datasets[datapoint_index][6].unsqueeze(0).to(device),  # mind the switched order
                   gold_energy=datasets[datapoint_index][5].unsqueeze(0).to(device),  # mind the switched order
                   utterance_embedding=datasets[datapoint_index][7].unsqueeze(0).to(device),
                   lang_ids=datasets[datapoint_index][8].unsqueeze(0).to(device),
                   return_mels=False).squeeze()
        if torch.isnan(loss):
            print(f"CAREFUL, NAN DETECTED: {datapoint_index}")
            losses.append(999999)
        else:
            losses.append(loss.item())
        index_pairs.append(datapoint_index)

    loss_high_to_low = sorted(losses, reverse=True)
    print(loss_high_to_low)
    threshold = loss_high_to_low[500]
    for index, loss in enumerate(losses):
        if loss > threshold:
            print(index_pairs[index])
            print(loss)
