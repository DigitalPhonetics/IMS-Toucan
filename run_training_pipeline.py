import argparse
import sys

from TrainingInterfaces.TrainingPipelines.FastSpeech2_MultiEnglish import run as fast_multi
from TrainingInterfaces.TrainingPipelines.FastSpeech2_Nancy import run as fast_nancy
from TrainingInterfaces.TrainingPipelines.HiFiGAN_combined import run as hifigan_combined
from TrainingInterfaces.TrainingPipelines.Tacotron2_MetaCheckpoint import run as meta_taco
from TrainingInterfaces.TrainingPipelines.Tacotron2_MultiEnglish import run as taco_multi
from TrainingInterfaces.TrainingPipelines.Tacotron2_Nancy import run as taco_nancy

pipeline_dict = {
    "fast_nancy": fast_nancy,
    "taco_nancy": taco_nancy,

    "hifi_combined": hifigan_combined,

    "taco_meta_checkpoint": meta_taco,

    "taco_multi": taco_multi,
    "fast_multi": fast_multi
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='IMS Toucan - Call to Train')

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
                        help="Whether to fine-tune from the specified checkpoint or continue training from it.",
                        default=False)

    parser.add_argument('--model_save_dir',
                        type=str,
                        help="Directory where the checkpoints should be saved to. A default should be specified in each individual pipeline.",
                        default=None)

    args = parser.parse_args()

    if args.finetune and args.resume_checkpoint is None:
        print("Need to provide path to checkpoint to fine-tune from!")
        sys.exit()

    if args.finetune and "hifi" in args.pipeline:
        print("Fine-tuning for HiFiGAN is not implemented as it didn't seem necessary. Should generalize across speakers without fine-tuning.")
        sys.exit()

    pipeline_dict[args.pipeline](gpu_id=args.gpu_id, resume_checkpoint=args.resume_checkpoint, finetune=args.finetune, model_dir=args.model_save_dir)
