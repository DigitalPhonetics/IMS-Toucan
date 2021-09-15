import argparse
import sys

from TrainingInterfaces.TrainingPipelines.FastSpeech2_LJSpeech import run as fast_lj
from TrainingInterfaces.TrainingPipelines.FastSpeech2_LibriTTS import run as fast_libri
from TrainingInterfaces.TrainingPipelines.FastSpeech2_Nancy import run as fast_nancy
from TrainingInterfaces.TrainingPipelines.FastSpeech2_Thorsten import run as fast_thorsten
from TrainingInterfaces.TrainingPipelines.HiFiGAN_combined import run as hifigan_combined
from TrainingInterfaces.TrainingPipelines.Tacotron2_LJSpeech import run as taco_lj
from TrainingInterfaces.TrainingPipelines.Tacotron2_LibriTTS import run as taco_libri
from TrainingInterfaces.TrainingPipelines.Tacotron2_MetaCheckpoint import run as meta_taco
from TrainingInterfaces.TrainingPipelines.Tacotron2_Nancy import run as taco_nancy
from TrainingInterfaces.TrainingPipelines.Tacotron2_Thorsten import run as taco_thorsten

pipeline_dict = {
    "fast_thorsten": fast_thorsten,
    "taco_thorsten": taco_thorsten,

    "fast_libri": fast_libri,
    "taco_libri": taco_libri,

    "fast_lj": fast_lj,
    "taco_lj": taco_lj,

    "fast_nancy": fast_nancy,
    "taco_nancy": taco_nancy,

    "hifi_combined": hifigan_combined,

    "meta_checkpoint": meta_taco
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
