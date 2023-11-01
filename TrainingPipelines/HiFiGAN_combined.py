import time

import wandb

from Architectures.Vocoder.HiFiGAN_Dataset import HiFiGANDataset
from Architectures.Vocoder.HiFiGAN_Discriminators import AvocodoHiFiGANJointDiscriminator
from Architectures.Vocoder.HiFiGAN_Generator import HiFiGAN
from Architectures.Vocoder.HiFiGAN_train_loop import train_loop
from Utility.path_to_transcript_dicts import *
from Utility.storage_config import MODELS_DIR


def run(gpu_id, resume_checkpoint, finetune, resume, model_dir, use_wandb, wandb_resume_id, gpu_count):
    if gpu_id == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    if gpu_count > 1:
        print("Multi GPU training not supported for HiFiGAN!")
        import sys
        sys.exit()

    print("Preparing")
    if model_dir is not None:
        model_save_dir = model_dir
    else:
        model_save_dir = os.path.join(MODELS_DIR, "HiFiGAN")
    os.makedirs(model_save_dir, exist_ok=True)

    print("Preparing new data...")
    file_lists_for_this_run_combined = list()
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_mls_italian().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_mls_english().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_gigaspeech().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_mls_french().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_mls_dutch().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_mls_polish().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_mls_spanish().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_mls_portuguese().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_karlsson().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_eva().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_bernd().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_friedrich().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_hokus().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_hui_others().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_elizabeth().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_nancy().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_vctk().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_libritts_all_clean().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_ljspeech().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_css10cmn().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_vietTTS().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_thorsten_emotional().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_thorsten_2022_10().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_thorsten_neutral().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_css10el().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_css10nl().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_css10fi().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_css10ru().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_css10hu().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_css10es().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_css10fr().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_nvidia_hifitts().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_spanish_blizzard_train().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_aishell3().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_VIVOS_viet().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_RAVDESS().keys())
    file_lists_for_this_run_combined += list(build_path_to_transcript_dict_ESDS().keys())
    file_lists_for_this_run_combined += list(build_file_list_singing_voice_audio_database())
    print("filepaths collected")

    fisher_yates_shuffle(file_lists_for_this_run_combined)
    fisher_yates_shuffle(file_lists_for_this_run_combined)
    fisher_yates_shuffle(file_lists_for_this_run_combined)
    print("filepaths randomized")

    train_set = HiFiGANDataset(list_of_paths=file_lists_for_this_run_combined[:250000])  # adjust the sample size until it fits into RAM

    generator = HiFiGAN()
    discriminator = AvocodoHiFiGANJointDiscriminator()

    print("Training model")
    if use_wandb:
        wandb.init(
            name=f"{__name__.split('.')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}" if wandb_resume_id is None else None,
            id=wandb_resume_id,  # this is None if not specified in the command line arguments.
            resume="must" if wandb_resume_id is not None else None)
    train_loop(batch_size=24,
               epochs=80000,
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
