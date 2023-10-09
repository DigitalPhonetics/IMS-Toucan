"""
This is basically an integration test
"""

import time

import torch
import wandb

from TTSTrainingInterfaces.ToucanTTS.ToucanTTS import ToucanTTS
from TTSTrainingInterfaces.ToucanTTS.toucantts_train_loop_arbiter import train_loop
from Utility.corpus_preparation import prepare_tts_corpus
from Utility.path_to_transcript_dicts import *
from Utility.storage_config import MODELS_DIR
from Utility.storage_config import PREPROCESSING_DIR


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume, use_wandb, wandb_resume_id, distributed):
    if gpu_id == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    print("Preparing")

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join(MODELS_DIR, "ToucanTTS_IntegrationTest")
    os.makedirs(save_dir, exist_ok=True)

    train_set = prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_integration_test(),
                                   corpus_dir=os.path.join(PREPROCESSING_DIR, "IntegrationTest"),
                                   lang="en",
                                   save_imgs=True)

    model = ToucanTTS()

    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        train_sampler = torch.utils.data.DistributedSampler(train_set, shuffle=True)
        model.to(local_rank)
        model = torch.utils.data.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        torch.distributed.barrier()
        # torch.cuda.synchronize()
    else:
        train_sampler = torch.utils.data.RandomSampler(train_set)

    if use_wandb:
        wandb.init(
            name=f"{__name__.split('.')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}" if wandb_resume_id is None else None,
            id=wandb_resume_id,  # this is None if not specified in the command line arguments.
            resume="must" if wandb_resume_id is not None else None)
    print("Training model")
    train_loop(net=model,
               datasets=[train_set],
               device=device,
               save_directory=save_dir,
               batch_size=8,
               eval_lang="en",
               warmup_steps=500,
               path_to_checkpoint=resume_checkpoint,
               path_to_embed_model=None,
               fine_tune=finetune,
               resume=resume,
               steps=5000,
               use_wandb=use_wandb,
               train_embed=True,
               train_samplers=[train_sampler])
    if use_wandb:
        wandb.finish()
