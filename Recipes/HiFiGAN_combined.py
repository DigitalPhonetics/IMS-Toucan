import time

import soundfile as sf
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
        model_save_dir = os.path.join(MODELS_DIR, "HiFiGAN_clean_data_and_augmentation")
    os.makedirs(model_save_dir, exist_ok=True)

    print("Preparing new data...")

    take_all = False  # use only files with a large enough samplerate or just use all of them

    file_lists_for_this_run_combined = list()
    fl = list(build_path_to_transcript_dict_mls_italian().keys())
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        file_lists_for_this_run_combined += fl

    fl = list(build_path_to_transcript_dict_mls_english().keys())
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        file_lists_for_this_run_combined += fl

    fl = list(build_path_to_transcript_dict_mls_french().keys())
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        file_lists_for_this_run_combined += fl

    fl = list(build_path_to_transcript_dict_mls_dutch().keys())
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        file_lists_for_this_run_combined += fl

    fl = list(build_path_to_transcript_dict_mls_polish().keys())
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        file_lists_for_this_run_combined += fl

    fl = list(build_path_to_transcript_dict_mls_spanish().keys())
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        file_lists_for_this_run_combined += fl

    fl = list(build_path_to_transcript_dict_mls_portuguese().keys())
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        file_lists_for_this_run_combined += fl

    fl = list(build_path_to_transcript_dict_karlsson().keys())
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        file_lists_for_this_run_combined += fl

    fl = list(build_path_to_transcript_dict_eva().keys())
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        file_lists_for_this_run_combined += fl

    fl = list(build_path_to_transcript_dict_bernd().keys())
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        file_lists_for_this_run_combined += fl

    fl = list(build_path_to_transcript_dict_friedrich().keys())
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        file_lists_for_this_run_combined += fl

    fl = list(build_path_to_transcript_dict_hokus().keys())
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        file_lists_for_this_run_combined += fl

    fl = list(build_path_to_transcript_dict_hui_others().keys())
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        file_lists_for_this_run_combined += fl

    fl = list(build_path_to_transcript_dict_elizabeth().keys())
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        file_lists_for_this_run_combined += fl

    fl = list(build_path_to_transcript_dict_nancy().keys())
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        file_lists_for_this_run_combined += fl

    fl = list(build_path_to_transcript_dict_vctk().keys())
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        file_lists_for_this_run_combined += fl

    fl = list(build_path_to_transcript_dict_libritts_all_clean().keys())
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        file_lists_for_this_run_combined += fl

    fl = list(build_path_to_transcript_dict_ljspeech().keys())
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        file_lists_for_this_run_combined += fl

    fl = list(build_path_to_transcript_dict_css10cmn().keys())
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        file_lists_for_this_run_combined += fl

    fl = list(build_path_to_transcript_dict_thorsten_emotional().keys())
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        file_lists_for_this_run_combined += fl

    fl = list(build_path_to_transcript_dict_thorsten_2022_10().keys())
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        file_lists_for_this_run_combined += fl

    fl = list(build_path_to_transcript_dict_thorsten_neutral().keys())
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        file_lists_for_this_run_combined += fl

    fl = list(build_path_to_transcript_dict_css10el().keys())
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        file_lists_for_this_run_combined += fl

    fl = list(build_path_to_transcript_dict_css10nl().keys())
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        file_lists_for_this_run_combined += fl
    fl = list(build_path_to_transcript_dict_css10fi().keys())
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        file_lists_for_this_run_combined += fl

    fl = list(build_path_to_transcript_dict_css10ru().keys())
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        file_lists_for_this_run_combined += fl

    fl = list(build_path_to_transcript_dict_css10hu().keys())
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        file_lists_for_this_run_combined += fl

    fl = list(build_path_to_transcript_dict_css10es().keys())
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        file_lists_for_this_run_combined += fl

    fl = list(build_path_to_transcript_dict_css10fr().keys())
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        file_lists_for_this_run_combined += fl

    fl = list(build_path_to_transcript_dict_nvidia_hifitts().keys())
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        file_lists_for_this_run_combined += fl

    fl = list(build_path_to_transcript_dict_spanish_blizzard_train().keys())
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        file_lists_for_this_run_combined += fl

    fl = list(build_path_to_transcript_dict_aishell3().keys())
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        file_lists_for_this_run_combined += fl

    fl = list(build_path_to_transcript_dict_VIVOS_viet().keys())
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        file_lists_for_this_run_combined += fl

    fl = list(build_path_to_transcript_dict_ESDS().keys())
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        file_lists_for_this_run_combined += fl

    fl = list(build_path_to_transcript_dict_CREMA_D().keys())
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        file_lists_for_this_run_combined += fl

    root = "/mount/arbeitsdaten61/studenten3/advanced-ml/2022/dhyanitr/projects/speech_datasets/Datasets/"
    for a in os.listdir(root):
        if os.path.isdir(a):
            for b in os.listdir(os.path.join(root, a)):
                if os.path.isdir(a):
                    for c in os.listdir(os.path.join(root, a)):
                        if os.path.isdir(a):
                            print("If you can read this, the directories were more nested than I thought.")
                        else:
                            if c.endswith(".wav") or c.endswith(".flac"):
                                _, sr = sf.read(os.path.join(root, a, b, c))
                                if sr >= 24000 or take_all:
                                    file_lists_for_this_run_combined.append(os.path.join(root, a, b, c))
                else:
                    if b.endswith(".wav") or b.endswith(".flac"):
                        _, sr = sf.read(os.path.join(root, a, b))
                        if sr >= 24000 or take_all:
                            file_lists_for_this_run_combined.append(os.path.join(root, a, b))
        else:
            if a.endswith(".wav") or a.endswith(".flac"):
                _, sr = sf.read(os.path.join(root, a))
                if sr >= 24000 or take_all:
                    file_lists_for_this_run_combined.append(os.path.join(root, a))

    print("filepaths collected")

    fisher_yates_shuffle(file_lists_for_this_run_combined)
    fisher_yates_shuffle(file_lists_for_this_run_combined)
    fisher_yates_shuffle(file_lists_for_this_run_combined)
    print("filepaths randomized")

    selection = file_lists_for_this_run_combined[:250000]  # adjust the sample size until it fits into RAM

    fl = list(build_path_to_transcript_dict_RAVDESS().keys())  # these three datasets are kind of important to represent some out-of-distribution data for what we expect.
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        selection += fl

    fl = build_file_list_singing_voice_audio_database()  # these three datasets are kind of important to represent some out-of-distribution data for what we expect.
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        selection += fl

    fl = list(build_path_to_transcript_dict_ears().keys())  # these three datasets are kind of important to represent some out-of-distribution data for what we expect.
    wav, sr = sf.read(fl[0])
    if sr >= 24000 or take_all:
        selection += fl

    fisher_yates_shuffle(selection)
    fisher_yates_shuffle(selection)

    train_set = HiFiGANDataset(list_of_paths=selection, use_random_corruption=False)

    generator = HiFiGAN()
    discriminator = AvocodoHiFiGANJointDiscriminator()

    print("Training model")
    if use_wandb:
        wandb.init(
            name=f"{__name__.split('.')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}" if wandb_resume_id is None else None,
            id=wandb_resume_id,  # this is None if not specified in the command line arguments.
            resume="must" if wandb_resume_id is not None else None)
    train_loop(batch_size=64,
               epochs=180000,
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
