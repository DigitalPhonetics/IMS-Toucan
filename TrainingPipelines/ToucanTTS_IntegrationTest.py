"""
This is basically an integration test
"""

import time

import torch
import torch.multiprocessing as mp
import wandb

from TTSTrainingInterfaces.ToucanTTS.ToucanTTS import ToucanTTS
from TTSTrainingInterfaces.ToucanTTS.toucantts_train_loop_arbiter import train_loop
from Utility.corpus_preparation import prepare_tts_corpus
from Utility.path_to_transcript_dicts import *
from Utility.storage_config import MODELS_DIR
from Utility.storage_config import PREPROCESSING_DIR


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume, use_wandb, wandb_resume_id, gpu_count):
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

    if gpu_count > 1:
        rank = int(os.environ["LOCAL_RANK"])
        queue = mp.SimpleQueue()
    else:
        rank = 0
    if rank == 0:
        train_set = prepare_tts_corpus(transcript_dict=build_path_to_transcript_dict_integration_test(),
                                       corpus_dir=os.path.join(PREPROCESSING_DIR, "IntegrationTest"),
                                       lang="en",
                                       save_imgs=True)
        if gpu_count > 1:
            print("Putting the dataset in shared memory")
            queue.put(train_set)

    if gpu_count > 1:
        torch.distributed.barrier()
        if rank > 0:
            # if this process is not the first process, we don't load the data, instead we wait until the first process is done loading it and then get it from the queue, so that everything is shared and not duplicated.
            print("retrieving the dataset from shared memory")
            train_set = queue.get()

    model = ToucanTTS()

    if gpu_count > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        train_sampler = torch.utils.data.DistributedSampler(train_set, shuffle=True)
        model.to(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(
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
        if gpu_count > 1:
            rank = int(os.environ["LOCAL_RANK"])
        else:
            rank = 0
        if rank == 0:
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
               train_samplers=[train_sampler],
               gpu_count=gpu_count)
    if use_wandb:
        wandb.finish()
