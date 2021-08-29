import argparse
import sys

import torch

from TrainingInterfaces.TrainingPipelines.FastSpeech2_LJSpeech import run as fast_LJSpeech
from TrainingInterfaces.TrainingPipelines.FastSpeech2_LibriTTS import run as fast_LibriTTS
from TrainingInterfaces.TrainingPipelines.FastSpeech2_Nancy import run as fast_Nancy
from TrainingInterfaces.TrainingPipelines.FastSpeech2_Thorsten import run as fast_Thorsten
from TrainingInterfaces.TrainingPipelines.HiFiGAN_combined import run as hifigan_combined
from TrainingInterfaces.TrainingPipelines.Tacotron2_LJSpeech import run as taco_LJSpeech
from TrainingInterfaces.TrainingPipelines.Tacotron2_LibriTTS import run as taco_LibriTTS
from TrainingInterfaces.TrainingPipelines.Tacotron2_Nancy import run as taco_Nancy
from TrainingInterfaces.TrainingPipelines.Tacotron2_Thorsten import run as taco_Thorsten

pipeline_dict = {
    "fast_Thorsten": fast_Thorsten,
    "taco_Thorsten": taco_Thorsten,

    "fast_LibriTTS": fast_LibriTTS,
    "taco_LibriTTS": taco_LibriTTS,

    "fast_LJSpeech": fast_LJSpeech,
    "taco_LJSpeech": taco_LJSpeech,

    "fast_Nancy"   : fast_Nancy,
    "taco_Nancy"   : taco_Nancy,

    "hifi_combined": hifigan_combined,
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

    if args.finetune and "hifigan" in args.pipeline:
        print("Fine-tuning for HiFiGAN is not implemented as it didn't seem necessary. Should generalize across speakers without fine-tuning.")
        sys.exit()

    if "fast" in args.pipeline:
        torch.multiprocessing.set_start_method('spawn', force=False)

    pipeline_dict[args.pipeline](gpu_id=args.gpu_id, resume_checkpoint=args.resume_checkpoint, finetune=args.finetune, model_dir=args.model_save_dir)
