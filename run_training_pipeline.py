import argparse
import sys

import torch

from TrainingInterfaces.TrainingPipelines.FastSpeech2_LJSpeech import run as fast_LJSpeech
from TrainingInterfaces.TrainingPipelines.FastSpeech2_LibriTTS import run as fast_LibriTTS
from TrainingInterfaces.TrainingPipelines.FastSpeech2_Nancy import run as fast_Nancy
from TrainingInterfaces.TrainingPipelines.FastSpeech2_Thorsten import run as fast_Thorsten
from TrainingInterfaces.TrainingPipelines.MelGAN_LJSpeech import run as melgan_LJSpeech
from TrainingInterfaces.TrainingPipelines.MelGAN_Nancy import run as melgan_Nancy
from TrainingInterfaces.TrainingPipelines.MelGAN_Thorsten import run as melgan_Thorsten
from TrainingInterfaces.TrainingPipelines.MelGAN_combined import run as melgan_combined
from TrainingInterfaces.TrainingPipelines.Tacotron2_LJSpeech import run as taco_LJSpeech
from TrainingInterfaces.TrainingPipelines.Tacotron2_LibriTTS import run as taco_LibriTTS
from TrainingInterfaces.TrainingPipelines.Tacotron2_Nancy import run as taco_Nancy
from TrainingInterfaces.TrainingPipelines.Tacotron2_Thorsten import run as taco_Thorsten

pipeline_dict = {
    "fast_Thorsten"  : fast_Thorsten,
    "melgan_Thorsten": melgan_Thorsten,
    "taco_Thorsten"  : taco_Thorsten,

    "fast_LibriTTS"  : fast_LibriTTS,
    "taco_LibriTTS"  : taco_LibriTTS,

    "fast_LJSpeech"  : fast_LJSpeech,
    "melgan_LJSpeech": melgan_LJSpeech,
    "taco_LJSpeech"  : taco_LJSpeech,

    "fast_Nancy"     : fast_Nancy,
    "melgan_Nancy"   : melgan_Nancy,
    "taco_Nancy"     : taco_Nancy,

    "melgan_combined": melgan_combined,
    }

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='IMS Speech Synthesis Toolkit - Call to Train')

    parser.add_argument('pipeline',
                        choices=list(pipeline_dict.keys()),
                        help="Select pipeline to train.")

    parser.add_argument('--gpu_id',
                        type=str,
                        help="Which GPU to run on. If not specified runs on CPU, but other than for integration tests that doesn't make much sense.",
                        default="cpu")

    parser.add_argument('--resume_checkpoint',
                        type=str,
                        help="Path to checkpoint to resume from.",
                        default=None)

    parser.add_argument('--finetune',
                        action="store_true",
                        help="Whether to fine-tune from the specified checkpoint.",
                        default=False)

    parser.add_argument('--model_save_dir',
                        type=str,
                        help="Directory where the checkpoints should be saved to.",
                        default=None)

    args = parser.parse_args()

    if args.finetune and args.resume_checkpoint is None:
        print("Need to provide path to checkpoint to fine-tune from!")
        sys.exit()

    if args.finetune and "melgan" in args.pipeline:
        print("Fine-tuning for MelGAN is not implemented as it didn't seem necessary and the GAN would most likely fail. Just train from scratch.")
        sys.exit()

    if "fast" in args.pipeline:
        torch.multiprocessing.set_start_method('spawn', force=False)

    pipeline_dict[args.pipeline](gpu_id=args.gpu_id, resume_checkpoint=args.resume_checkpoint, finetune=args.finetune, model_dir=args.model_save_dir)
