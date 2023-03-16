import argparse
import sys

from TrainingInterfaces.TrainingPipelines.Corpora_extract_phonemes_from_labelfiles import run as extract_lab
from TrainingInterfaces.TrainingPipelines.FastSpeech2_German_Austrian import run as fast_austrian
from TrainingInterfaces.TrainingPipelines.FastSpeech2_German_Austrian_lang_emb import run as lang_emb
from TrainingInterfaces.TrainingPipelines.FastSpeech2_German_Austrian_avg_lang_emb import run as avg_lang_emb
from TrainingInterfaces.TrainingPipelines.FastSpeech2_multilingual_lang_emb import run as multi_lang_emb
from TrainingInterfaces.TrainingPipelines.FastSpeech2_IntegrationTest import run as integration_test
from TrainingInterfaces.TrainingPipelines.FastSpeech2_IntegrationTestVietnamese import run as integration_test_vietnamese
from TrainingInterfaces.TrainingPipelines.FastSpeech2_MetaCheckpoint import run as meta_fast
from TrainingInterfaces.TrainingPipelines.FastSpeech2_finetune_to_German import run as fine_ger
from TrainingInterfaces.TrainingPipelines.HiFiGAN_combined import run as hifigan_combined
from TrainingInterfaces.TrainingPipelines.HiFiGAN_aridialect import run as hifigan_aridialect
from TrainingInterfaces.TrainingPipelines.pretrain_aligner import run as aligner

pipeline_dict = {
    "meta": meta_fast,
    "hifi_combined": hifigan_combined,
    "hifi_aridialect": hifigan_aridialect,
    "extract_lab"   : extract_lab,
    "fast_austrian" : fast_austrian,
    "aridialect_lang_emb" : lang_emb, 
    "aridialect_avg_lang_emb" : avg_lang_emb,
    "aridialect_multi_avg_lang_emb" : multi_lang_emb,
    "aligner": aligner,
    "fine_ger": fine_ger,
    "integration_test": integration_test,
    "integration_test_vietnamese": integration_test_vietnamese
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
