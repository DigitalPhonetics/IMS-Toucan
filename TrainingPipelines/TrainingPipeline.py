import time

import wandb

from Architectures.ToucanTTS.ToucanTTS import ToucanTTS
from Architectures.ToucanTTS.toucantts_train_loop import train_loop
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
        save_dir = os.path.join(MODELS_DIR, "ToucanTTS_LatentToSpectrogram")
    os.makedirs(save_dir, exist_ok=True)

    if gpu_count > 1:
        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(backend="nccl")
    else:
        rank = 0

    train_set = prepare_tts_corpus(path_list=list(),  # TODO
                                   latents_list=list(),  # TODO
                                   corpus_dir=os.path.join(PREPROCESSING_DIR, "LatentToSpectrogram"),
                                   gpu_count=gpu_count,
                                   rank=rank)

    model = ToucanTTS()

    if gpu_count > 1:
        model.to(rank)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True,
        )
        torch.distributed.barrier()
    train_sampler = torch.utils.data.RandomSampler(train_set)

    if use_wandb:
        if rank == 0:
            wandb.init(
                name=f"{__name__.split('.')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}" if wandb_resume_id is None else None,
                id=wandb_resume_id,  # this is None if not specified in the command line arguments.
                resume="must" if wandb_resume_id is not None else None)
    print("Training model")
    train_loop(net=model,
               train_dataset=train_set,
               device=device,
               save_directory=save_dir,
               path_to_checkpoint=resume_checkpoint,
               fine_tune=finetune,
               resume=resume,
               use_wandb=use_wandb,
               train_sampler=train_sampler,
               gpu_count=gpu_count,
               batch_size=12,
               lr=0.001,
               warmup_steps=8000,
               steps=100000,
               steps_per_checkpoint=1000
               )
    if use_wandb:
        wandb.finish()
