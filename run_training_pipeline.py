import argparse
import os
import random
import sys

import torch

from TrainingInterfaces.TrainingPipelines.Avocodo_combined import run as hifi_codo
from TrainingInterfaces.TrainingPipelines.BigVGAN_combined import run as bigvgan
from TrainingInterfaces.TrainingPipelines.FastSpeech2Embedding_IntegrationTest import run as fs_integration_test
from TrainingInterfaces.TrainingPipelines.GST_FastSpeech2 import run as embedding
from TrainingInterfaces.TrainingPipelines.StochasticToucanTTS_Nancy import run as nancystoch
from TrainingInterfaces.TrainingPipelines.ToucanTTS_IntegrationTest import run as tt_integration_test
from TrainingInterfaces.TrainingPipelines.ToucanTTS_MetaCheckpoint import run as meta
from TrainingInterfaces.TrainingPipelines.ToucanTTS_Nancy import run as nancy
from TrainingInterfaces.TrainingPipelines.finetuning_example import run as fine_tuning_example
from TrainingInterfaces.TrainingPipelines.pretrain_aligner import run as aligner

pipeline_dict = {
    # the finetuning example
    "fine_ex"       : fine_tuning_example,
    # integration tests
    "fs_it"         : fs_integration_test,
    "tt_it"         : tt_integration_test,
    # regular ToucanTTS pipelines
    "nancy"         : nancy,
    "nancystoch"    : nancystoch,
    "meta"          : meta,
    # training vocoders (not recommended, best to use provided checkpoint)
    "avocodo"       : hifi_codo,
    "bigvgan"       : bigvgan,
    # training the GST embedding jointly with FastSpeech 2 on expressive data (not recommended, best to use provided checkpoint)
    "embedding"     : embedding,
    # training the aligner from scratch (not recommended, best to use provided checkpoint)
    "aligner"       : aligner,
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training with the IMS Toucan Speech Synthesis Toolkit')

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
                        help="Whether to use weights and biases to track training runs. Requires you to run wandb login and place your auth key before.",
                        default=False)

    parser.add_argument('--wandb_resume_id',
                        type=str,
                        help="ID of a stopped wandb run to continue tracking",
                        default=None)

    args = parser.parse_args()

    if args.finetune and args.resume_checkpoint is None and not args.resume:
        print("Need to provide path to checkpoint to fine-tune from!")
        sys.exit()

    if args.gpu_id == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")
        print(f"No GPU specified, using CPU. Training will likely not work without GPU.")

    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"
        device = torch.device("cuda")
        print(f"Making GPU {os.environ['CUDA_VISIBLE_DEVICES']} the only visible device.")

    torch.manual_seed(131714)
    random.seed(131714)
    torch.random.manual_seed(131714)

    pipeline_dict[args.pipeline](gpu_id=args.gpu_id,
                                 resume_checkpoint=args.resume_checkpoint,
                                 resume=args.resume,
                                 finetune=args.finetune,
                                 model_dir=args.model_save_dir,
                                 use_wandb=args.wandb,
                                 wandb_resume_id=args.wandb_resume_id)
