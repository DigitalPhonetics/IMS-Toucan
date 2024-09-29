import time

import torch
import wandb

from Modules.Vocoder.BigVGAN import BigVGAN
from Modules.Vocoder.HiFiGAN_Discriminators import AvocodoHiFiGANJointDiscriminator
from Modules.Vocoder.HiFiGAN_E2E_Dataset import HiFiGANDataset
from Modules.Vocoder.HiFiGAN_train_loop import train_loop
from Utility.path_to_transcript_dicts import *
from Utility.storage_config import MODELS_DIR


def run(gpu_id, resume_checkpoint, finetune, resume, model_dir, use_wandb, wandb_resume_id, gpu_count):
    if gpu_id == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    if gpu_count > 1:
        print("Multi GPU training not supported for BigVGAN!")
        import sys
        sys.exit()

    print("Preparing")
    if model_dir is not None:
        model_save_dir = model_dir
    else:
        model_save_dir = os.path.join(MODELS_DIR, "BigVGAN_e2e")
    os.makedirs(model_save_dir, exist_ok=True)

    # To prepare the data, have a look at Modules/Vocoder/run_end-to-end_data_creation

    print("Collecting new data...")

    file_lists_for_this_run_combined = list()
    file_lists_for_this_run_combined_synthetic = list()

    fl = list(build_path_to_transcript_libritts_all_clean().keys())
    fisher_yates_shuffle(fl)
    fisher_yates_shuffle(fl)
    for i, f in enumerate(fl):
        if os.path.exists(f.replace(".wav", "_synthetic_spec.pt")):
            file_lists_for_this_run_combined.append(f)
            file_lists_for_this_run_combined_synthetic.append(f.replace(".wav", "_synthetic_spec.pt"))
    print("filepaths collected")

    train_set = HiFiGANDataset(list_of_original_paths=file_lists_for_this_run_combined,
                               list_of_synthetic_paths=file_lists_for_this_run_combined_synthetic)

    generator = BigVGAN()
    discriminator = AvocodoHiFiGANJointDiscriminator()

    print("Training model")
    if use_wandb:
        wandb.init(
            name=f"{__name__.split('.')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}" if wandb_resume_id is None else None,
            id=wandb_resume_id,  # this is None if not specified in the command line arguments.
            resume="must" if wandb_resume_id is not None else None)
    train_loop(batch_size=16,
               epochs=5180000,
               generator=generator,
               discriminator=discriminator,
               train_dataset=train_set,
               device=device,
               epochs_per_save=1,
               model_save_dir=model_save_dir,
               path_to_checkpoint=resume_checkpoint,
               resume=resume,
               use_wandb=use_wandb,
               finetune=finetune)
    if use_wandb:
        wandb.finish()


def fisher_yates_shuffle(lst):
    for i in range(len(lst) - 1, 0, -1):
        j = random.randint(0, i)
        lst[i], lst[j] = lst[j], lst[i]
