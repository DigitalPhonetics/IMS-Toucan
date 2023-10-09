import argparse
import os
import random
import sys

import torch

from TrainingPipelines.AlignerPipeline import run as aligner
from TrainingPipelines.StochasticToucanTTS_Nancy import run as nancystoch
from TrainingPipelines.ToucanTTS_IntegrationTest import run as tt_integration_test
from TrainingPipelines.ToucanTTS_MLS_English import run as mls
from TrainingPipelines.ToucanTTS_MetaCheckpoint import run as meta
from TrainingPipelines.ToucanTTS_Nancy import run as nancy
from TrainingPipelines.finetuning_example import run as fine_tuning_example

pipeline_dict = {
    # the finetuning example
    "finetuning_example": fine_tuning_example,
    # integration tests
    "tt_it"             : tt_integration_test,
    # regular ToucanTTS pipelines
    "nancy"             : nancy,
    "mls"               : mls,
    "nancystoch"        : nancystoch,
    "meta"              : meta,
    # training the aligner from scratch (not recommended, best to use provided checkpoint)
    "aligner"           : aligner,
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training with the IMS Toucan Speech Synthesis Toolkit')

    parser.add_argument('pipeline',
                        choices=list(pipeline_dict.keys()),
                        help="Select pipeline to train.")

    parser.add_argument('--gpu_id',
                        type=str,
                        help="Which GPU(s) to run on. If not specified runs on CPU, but other than for integration tests that doesn't make much sense.",
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

    parser.add_argument('--distributed',
                        action="store_true",
                        help="Whether to use distributed training.",
                        default=False)

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

    if args.distributed:
        print("Running this job across all specified GPUs. Make sure to start this run with torchrun to get the benefits of torch elastic! It might not work otherwise.")
        # torchrun --standalone --nproc_per_node=4 --nnodes=1 run_training_pipeline.py --distributed --gpu_id 1,2,3

    torch.manual_seed(9665)
    random.seed(9665)
    torch.random.manual_seed(9665)

    torch.multiprocessing.set_sharing_strategy('file_system')

    pipeline_dict[args.pipeline](gpu_id=args.gpu_id,
                                 resume_checkpoint=args.resume_checkpoint,
                                 resume=args.resume,
                                 finetune=args.finetune,
                                 model_dir=args.model_save_dir,
                                 use_wandb=args.wandb,
                                 wandb_resume_id=args.wandb_resume_id,
                                 distributed=args.distributed)
