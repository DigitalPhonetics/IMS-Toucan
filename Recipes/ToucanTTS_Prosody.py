"""
Script to train with probabilistic variance prediction

Comments in ALL CAPS are instructions
"""

import time

import wandb
from torch.utils.data import ConcatDataset

from Modules.ToucanTTS.ToucanTTS import ToucanTTS
from Modules.ToucanTTS.ToucanTTS_nf import ToucanTTS_nf
from Modules.Toucan_det.ToucanTTS import ToucanTTS as ToucanTTS_det
from Modules.ToucanTTS.toucantts_train_loop_arbiter import train_loop
from Utility.corpus_preparation import prepare_tts_corpus
from Utility.path_to_transcript_dicts import *
from Utility.storage_config import MODELS_DIR
from Utility.storage_config import PREPROCESSING_DIR


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume, use_wandb, wandb_resume_id, gpu_count):
    if gpu_id == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    assert gpu_count == 1  # distributed finetuning is not supported

    # IF YOU'RE ADDING A NEW LANGUAGE, YOU MIGHT NEED TO ADD HANDLING FOR IT IN Preprocessing/TextFrontend.py

    print("Preparing")

    # Parameter for the probabilstic model
    order = "ped" # pitch, energy, duration also epd and all (in one) possible
    prosody_channels = 8
    predictor_layers = 3
    predictor_kernel_size = 5
    predictor_dropout_rate = 0.2
    architecture = "CFM" # "NF"
    start_reflow = 91000 # if set higher than steps will not be used!
    dropout = False
    log = False

    # create savepath
    save_path = f"{architecture}_{order}"
    if log:
        save_path += "_log"
    if dropout:
        save_path += "_drop_01"
    if architecture == "RF":
            save_path += f"re_{start_reflow}" 
    save_path += f"_c{prosody_channels}_l{predictor_layers}_k{predictor_kernel_size}_d{predictor_dropout_rate}"

    print("Config: ")
    print(f"order: {order} path: {save_path} channels: {prosody_channels}, drop: {dropout}, log: {log}")

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join(MODELS_DIR, save_path)  # RENAME TO SOMETHING MEANINGFUL FOR YOUR DATA
    os.makedirs(save_dir, exist_ok=True)

    train_data = prepare_tts_corpus(transcript_dict=build_path_to_transcript_libritts_all_clean(),
                                   corpus_dir=os.path.join(PREPROCESSING_DIR, "libri"),
                                    lang="eng")  # CHANGE THE TRANSCRIPT DICT, THE NAME OF THE CACHE DIRECTORY AND THE LANGUAGE TO YOUR NEEDS

    if architecture == "CFM" or architecture == "RF":
        model = ToucanTTS(prosody_order=order, prosody_channels=prosody_channels, dropout=dropout, duration_log_scale=log,
                      duration_predictor_layers=predictor_layers, pitch_predictor_layers=predictor_layers, energy_predictor_layers=predictor_layers,
                      duration_predictor_kernel_size=predictor_kernel_size, pitch_predictor_kernel_size=predictor_kernel_size, energy_predictor_kernel_size=predictor_kernel_size,
                      duration_predictor_dropout_rate=predictor_dropout_rate, pitch_predictor_dropout=predictor_dropout_rate, energy_predictor_dropout=predictor_dropout_rate)
    elif architecture == "NF":
        model = ToucanTTS_nf(prosody_order=order, prosody_channels=prosody_channels, dropout=dropout, duration_log_scale=log,
                      duration_predictor_layers=predictor_layers, pitch_predictor_layers=predictor_layers, energy_predictor_layers=2, ## Set manually!!!!!!!!!!
                      duration_predictor_kernel_size=predictor_kernel_size, pitch_predictor_kernel_size=predictor_kernel_size, energy_predictor_kernel_size=3,
                      duration_predictor_dropout_rate=predictor_dropout_rate, pitch_predictor_dropout=predictor_dropout_rate, energy_predictor_dropout=predictor_dropout_rate)
    elif architecture == "DET":
        model = ToucanTTS_det(prosody_order=order, prosody_channels=prosody_channels, dropout=dropout, duration_log_scale=log,
                      duration_predictor_layers=predictor_layers, pitch_predictor_layers=predictor_layers, energy_predictor_layers=predictor_layers,
                      duration_predictor_kernel_size=predictor_kernel_size, pitch_predictor_kernel_size=predictor_kernel_size, energy_predictor_kernel_size=predictor_kernel_size,
                      duration_predictor_dropout_rate=predictor_dropout_rate, pitch_predictor_dropout=predictor_dropout_rate, energy_predictor_dropout=predictor_dropout_rate)

    if use_wandb:
        name = save_path.split("/")[-1] + "_" + time.strftime('%Y%m%d-%H%M%S')
        wandb.init(
            name=f"{name}" if wandb_resume_id is None else None,
            id=wandb_resume_id,  # this is None if not specified in the command line arguments.
            resume="must" if wandb_resume_id is not None else None)

    print("Training model")
    train_loop(net=model,
            datasets=[train_data],
            device=device,
            save_directory=save_dir,
            batch_size=4,  # YOU MIGHT GET OUT OF MEMORY ISSUES ON SMALL GPUs, IF SO, DECREASE THIS.
            eval_lang="eng",  # THE LANGUAGE YOUR PROGRESS PLOTS WILL BE MADE IN
            warmup_steps=5000, #5000
            lr=1e-4,  # if you have enough data (over ~1000 datapoints) you can increase this up to 1e-4 and it will still be stable, but learn quicker.
            # DOWNLOAD THESE INITIALIZATION MODELS FROM THE RELEASE PAGE OF THE GITHUB OR RUN THE DOWNLOADER SCRIPT TO GET THEM AUTOMATICALLY
            path_to_checkpoint=None, #os.path.join(MODELS_DIR, "ToucanTTS_Meta", "best.pt") if resume_checkpoint is None else resume_checkpoint,
            fine_tune=True if resume_checkpoint is None and not resume else finetune,
            resume=resume,
            steps=110000,
            steps_per_checkpoint=1000, #1000
            use_wandb=use_wandb,
            train_samplers=[torch.utils.data.RandomSampler(train_data)],
            gpu_count=1,
            architecture=architecture,
            start_reflow=start_reflow)
    if use_wandb:
        wandb.finish()
