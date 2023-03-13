import argparse
import sys

from TrainingInterfaces.TrainingPipelines.BigVGAN_combined import run as bigvgan
from TrainingInterfaces.TrainingPipelines.FastSpeech2_IntegrationTest import run as fs_integration_test
from TrainingInterfaces.TrainingPipelines.HiFiGAN_Avocodo import run as hifi_codo
from TrainingInterfaces.TrainingPipelines.JointEmbeddingFunction import run as embedding
from TrainingInterfaces.TrainingPipelines.PortaSpeech_AD import run as ad
from TrainingInterfaces.TrainingPipelines.PortaSpeech_French import run as french
from TrainingInterfaces.TrainingPipelines.PortaSpeech_IntegrationTest import run as ps_integration_test
from TrainingInterfaces.TrainingPipelines.PortaSpeech_NEB import run as neb
from TrainingInterfaces.TrainingPipelines.ToucanTTS_Libri import run as libri
from TrainingInterfaces.TrainingPipelines.ToucanTTS_MetaCheckpoint import run as meta
from TrainingInterfaces.TrainingPipelines.ToucanTTS_NEB import run as toucanneb
from TrainingInterfaces.TrainingPipelines.ToucanTTS_Nancy import run as nancy
from TrainingInterfaces.TrainingPipelines.finetuning_example import run as fine_tuning_example
from TrainingInterfaces.TrainingPipelines.pretrain_aligner import run as aligner

pipeline_dict = {
    "meta"     : meta,
    "embedding": embedding,
    "hifi_codo": hifi_codo,
    "aligner"  : aligner,
    "fine_ex"  : fine_tuning_example,
    "fs_it"    : fs_integration_test,
    "ps_it"    : ps_integration_test,
    "nancy"    : nancy,
    "ad"       : ad,
    "neb"      : neb,
    "toucanneb": toucanneb,
    "french"   : french,
    "libri"    : libri,
    "bigvgan"  : bigvgan
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

    parser.add_argument('--wandb',
                        action="store_true",
                        help="Whether to use weigths and biases to track training runs. Requires you to run wandb login and place your auth key before.",
                        default=False)

    parser.add_argument('--wandb_resume_id',
                        type=str,
                        help="ID of a stopped wandb run to continue tracking",
                        default=None)

    args = parser.parse_args()

    if args.finetune and args.resume_checkpoint is None and not args.resume:
        print("Need to provide path to checkpoint to fine-tune from!")
        sys.exit()

    pipeline_dict[args.pipeline](gpu_id=args.gpu_id,
                                 resume_checkpoint=args.resume_checkpoint,
                                 resume=args.resume,
                                 finetune=args.finetune,
                                 model_dir=args.model_save_dir,
                                 use_wandb=args.wandb,
                                 wandb_resume_id=args.wandb_resume_id)
