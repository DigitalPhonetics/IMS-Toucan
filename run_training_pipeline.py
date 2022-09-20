import argparse
import sys

from TrainingInterfaces.Spectrogram_to_Embedding.finetune_embeddings_to_tasks import finetune_model_emotion
from TrainingInterfaces.Spectrogram_to_Embedding.finetune_embeddings_to_tasks import finetune_model_speaker
from TrainingInterfaces.TrainingPipelines.FastSpeech2_Controllable import run as control
from TrainingInterfaces.TrainingPipelines.FastSpeech2_EmoGST import run as gst
from TrainingInterfaces.TrainingPipelines.FastSpeech2_IntegrationTest import run as integration_test
from TrainingInterfaces.TrainingPipelines.FastSpeech2_MetaCheckpoint import run as meta_fast
from TrainingInterfaces.TrainingPipelines.FastSpeech2_finetuning_example import run as fine_ger
from TrainingInterfaces.TrainingPipelines.HiFiGAN_Avocodo import run as hifigan_combined
from TrainingInterfaces.TrainingPipelines.pretrain_aligner import run as aligner

pipeline_dict = {
    "meta"            : meta_fast,
    "hifi_combined"   : hifigan_combined,
    "aligner"         : aligner,
    "fine_ger"        : fine_ger,
    "integration_test": integration_test,
    "gst"             : gst,
    "spk"             : finetune_model_speaker,
    "emo"             : finetune_model_emotion,
    "control"         : control,
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

    parser.add_argument('--resume',
                        action="store_true",
                        help="Automatically load the highest checkpoint and continue from there.",
                        default=False)

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

    pipeline_dict[args.pipeline](gpu_id=args.gpu_id,
                                 resume_checkpoint=args.resume_checkpoint,
                                 resume=args.resume,
                                 finetune=args.finetune,
                                 model_dir=args.model_save_dir)
