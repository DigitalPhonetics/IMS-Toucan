import argparse
import os
import random
import sys

import torch

from TrainingInterfaces.TrainingPipelines.Avocodo_combined import run as hifi_codo
from TrainingInterfaces.TrainingPipelines.BigVGAN_combined import run as bigvgan
from TrainingInterfaces.TrainingPipelines.FastSpeech2Embedding_IntegrationTest import run as fs_integration_test
from TrainingInterfaces.TrainingPipelines.GST_FastSpeech2 import run as embedding
from TrainingInterfaces.TrainingPipelines.GST_Blizzard2013 import run as b_embedding
from TrainingInterfaces.TrainingPipelines.GST_EmoVDBSam import run as emo_embedding
from TrainingInterfaces.TrainingPipelines.GST_EmoMulti import run as emomulti_embedding
from TrainingInterfaces.TrainingPipelines.StochasticToucanTTS_Nancy import run as nancystoch
from TrainingInterfaces.TrainingPipelines.ToucanTTS_IntegrationTest import run as tt_integration_test
from TrainingInterfaces.TrainingPipelines.ToucanTTS_MetaCheckpoint import run as meta
from TrainingInterfaces.TrainingPipelines.ToucanTTS_Nancy import run as nancy
from TrainingInterfaces.TrainingPipelines.ToucanTTS_Blizzard2013 import run as blizzard2013
from TrainingInterfaces.TrainingPipelines.ToucanTTS_Blizzard2013_sent_emb import run as blizzard2013_sent
from TrainingInterfaces.TrainingPipelines.ToucanTTS_Blizzard2013_word_emb import run as blizzard2013_word
from TrainingInterfaces.TrainingPipelines.ToucanTTS_Blizzard2013_sent_word_emb import run as blizzard2013_sent_word
from TrainingInterfaces.TrainingPipelines.ToucanTTS_EmoVDBSam import run as emo
from TrainingInterfaces.TrainingPipelines.ToucanTTS_EmoVDBSam_sent_emb import run as emo_sent
from TrainingInterfaces.TrainingPipelines.ToucanTTS_EmoVDBSam_word_emb import run as emo_word
from TrainingInterfaces.TrainingPipelines.ToucanTTS_EmoVDB import run as emovdb
from TrainingInterfaces.TrainingPipelines.ToucanTTS_EmoVDB_sent_emb import run as emovdb_sent
from TrainingInterfaces.TrainingPipelines.ToucanTTS_CremaD import run as cremad
from TrainingInterfaces.TrainingPipelines.ToucanTTS_CremaD_sent_emb import run as cremad_sent
from TrainingInterfaces.TrainingPipelines.ToucanTTS_Ravdess import run as ravdess
from TrainingInterfaces.TrainingPipelines.ToucanTTS_Ravdess_sent_emb import run as ravdess_sent
from TrainingInterfaces.TrainingPipelines.ToucanTTS_ESDS import run as esds
from TrainingInterfaces.TrainingPipelines.ToucanTTS_ESDS_sent_emb import run as esds_sent
from TrainingInterfaces.TrainingPipelines.ToucanTTS_EmoMulti import run as emomulti
from TrainingInterfaces.TrainingPipelines.ToucanTTS_EmoMulti_sent_emb import run as emomulti_sent
from TrainingInterfaces.TrainingPipelines.ToucanTTS_EmoMulti_sent_word_emb import run as emomulti_sent_word
from TrainingInterfaces.TrainingPipelines.ToucanTTS_EmoMulti_sent_emb_pretrain import run as emomulti_sent_pre
from TrainingInterfaces.TrainingPipelines.ToucanTTS_PromptSpeech import run as promptspeech
from TrainingInterfaces.TrainingPipelines.ToucanTTS_PromptSpeech_sent_emb import run as promptspeech_sent
from TrainingInterfaces.TrainingPipelines.ToucanTTS_LibriTTS import run as libri
from TrainingInterfaces.TrainingPipelines.ToucanTTS_LibriTTS_sent_emb import run as libri_sent
from TrainingInterfaces.TrainingPipelines.ToucanTTS_LJSpeech import run as lj
from TrainingInterfaces.TrainingPipelines.ToucanTTS_LJSpeech_word_emb import run as lj_word
from TrainingInterfaces.TrainingPipelines.finetuning_example import run as fine_tuning_example
from TrainingInterfaces.TrainingPipelines.pretrain_aligner import run as aligner
from TrainingInterfaces.TrainingPipelines.SentEmbAdaptor_Blizzard2013 import run as adapt
from TrainingInterfaces.TrainingPipelines.SentEmbAdaptor_EmoVDBSam import run as adapt_emo
from TrainingInterfaces.TrainingPipelines.SentEmbAdaptor_EmoMulti import run as adapt_emomulti

pipeline_dict = {
    # the finetuning example
    "fine_ex"       : fine_tuning_example,
    # integration tests
    "fs_it"         : fs_integration_test,
    "tt_it"         : tt_integration_test,
    # regular ToucanTTS pipelines
    "nancy"     : nancy,
    "nancystoch": nancystoch,
    "meta"      : meta,
    "blizzard2013": blizzard2013,
    "blizzard2013_sent": blizzard2013_sent,
    "blizzard2013_word": blizzard2013_word,
    "blizzard2013_sent_word" : blizzard2013_sent_word,
    "promptspeech": promptspeech,
    "promptspeech_sent": promptspeech_sent,
    "libri"         : libri,
    "libri_sent"    : libri_sent,
    "emo"           : emo,
    "emo_sent"      : emo_sent,
    "emo_word"      : emo_word,
    "lj"            : lj,
    "lj_word"       : lj_word,
    "emovdb"        : emovdb,
    "emovdb_sent"   : emovdb_sent,
    "cremad"        : cremad,
    "cremad_sent"   : cremad_sent,
    "ravdess"       : ravdess,
    "ravdess_sent"  : ravdess_sent,
    "esds"          : esds,
    "esds_sent"     : esds_sent,
    "emomulti"      : emomulti,
    "emomulti_sent"      : emomulti_sent,
    "emomulti_sent_word" : emomulti_sent_word,
    "emomulti_sent_pre" : emomulti_sent_pre,
    # training vocoders (not recommended, best to use provided checkpoint)
    "avocodo"       : hifi_codo,
    "bigvgan"       : bigvgan,
    # training the GST embedding jointly with FastSpeech 2 on expressive data (not recommended, best to use provided checkpoint)
    "embedding"     : embedding,
    "b_embedding"   : b_embedding,
    "emo_embedding" : emo_embedding,
    "emomulti_embedding" : emomulti_embedding,
    # training the aligner from scratch (not recommended, best to use provided checkpoint)
    "aligner"       : aligner,
    # training the sentence embedding adaptor
    "adapt"         : adapt,
    "adapt_emo"     : adapt_emo,
    "adapt_emomulti"     : adapt_emomulti,
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
