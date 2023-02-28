import time

import torch
import torch.multiprocessing
import wandb
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from TrainingInterfaces.Spectrogram_to_Embedding.StyleEmbedding import StyleEmbedding
from TrainingInterfaces.Text_to_Spectrogram.AutoAligner.AlignerDatasetBuilder import AlignerDatasetBuilder
from TrainingInterfaces.Text_to_Spectrogram.PortaSpeech.PortaSpeech import PortaSpeech
from TrainingInterfaces.Text_to_Spectrogram.PortaSpeech.portaspeech_train_loop_arbiter import train_loop
from Utility.corpus_preparation import prepare_fastspeech_corpus
from Utility.path_to_transcript_dicts import *
from Utility.storage_config import MODELS_DIR
from Utility.storage_config import PREPROCESSING_DIR


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume, use_wandb, wandb_resume_id, remove_faulty_samples=False,
        generate_all_aligner_caches=False):
    # It is not recommended training this yourself or to finetune this, but you can.
    # The recommended use is to download the pretrained model from the github release
    # page and finetune to your desired data

    torch.manual_seed(131714)
    random.seed(131714)
    torch.random.manual_seed(131714)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"

    datasets = list()

    base_dir = os.path.join(MODELS_DIR, "PortaSpeech_Meta")
    if model_dir is not None:
        meta_save_dir = model_dir
    else:
        meta_save_dir = base_dir
    os.makedirs(meta_save_dir, exist_ok=True)

    print("Preparing")

    if generate_all_aligner_caches:
        build_all_aligner_dataset_caches(torch.device("cuda"))

    english_datasets = list()
    german_datasets = list()
    greek_datasets = list()
    spanish_datasets = list()
    finnish_datasets = list()
    russian_datasets = list()
    hungarian_datasets = list()
    dutch_datasets = list()
    french_datasets = list()
    portuguese_datasets = list()
    polish_datasets = list()
    italian_datasets = list()
    chinese_datasets = list()
    vietnamese_datasets = list()

    english_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_nancy(),
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "Nancy"),
                                                      lang="en"))

    english_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_ljspeech(),
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "LJSpeech"),
                                                      lang="en"))

    english_datasets.append(
        prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_libritts_all_clean(),
                                  corpus_dir=os.path.join(PREPROCESSING_DIR, "libri_all_clean"),
                                  lang="en"))

    english_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_vctk(),
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "vctk"),
                                                      lang="en"))

    english_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_nvidia_hifitts(),
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "hifi"),
                                                      lang="en"))

    english_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_RAVDESS(),
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "ravdess"),
                                                      lang="en",
                                                      ctc_selection=False))

    english_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_ESDS(),
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "esds"),
                                                      lang="en"))

    german_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_karlsson(),
                                                     corpus_dir=os.path.join(PREPROCESSING_DIR, "Karlsson"),
                                                     lang="de"))

    german_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_eva(),
                                                     corpus_dir=os.path.join(PREPROCESSING_DIR, "Eva"),
                                                     lang="de"))

    german_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_hokus(),
                                                     corpus_dir=os.path.join(PREPROCESSING_DIR, "Hokus"),
                                                     lang="de"))

    german_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_bernd(),
                                                     corpus_dir=os.path.join(PREPROCESSING_DIR, "Bernd"),
                                                     lang="de"))

    german_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_hui_others(),
                                                     corpus_dir=os.path.join(PREPROCESSING_DIR, "hui_others"),
                                                     lang="de"))

    german_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_thorsten(),
                                                     corpus_dir=os.path.join(PREPROCESSING_DIR, "Thorsten"),
                                                     lang="de"))

    greek_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_css10el(),
                                                    corpus_dir=os.path.join(PREPROCESSING_DIR, "meta_Greek"),
                                                    lang="el"))

    spanish_datasets.append(
        prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_spanish_blizzard_train(),
                                  corpus_dir=os.path.join(PREPROCESSING_DIR, "spanish_blizzard"),
                                  lang="es"))

    spanish_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_css10es(),
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "meta_Spanish"),
                                                      lang="es"))

    spanish_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_mls_spanish(),
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_spanish"),
                                                      lang="es"))

    finnish_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_css10fi(),
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "meta_Finnish"),
                                                      lang="fi"))

    russian_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_css10ru(),
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "meta_Russian"),
                                                      lang="ru"))

    hungarian_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_css10hu(),
                                                        corpus_dir=os.path.join(PREPROCESSING_DIR, "meta_Hungarian"),
                                                        lang="hu"))

    dutch_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_css10nl(),
                                                    corpus_dir=os.path.join(PREPROCESSING_DIR, "meta_Dutch"),
                                                    lang="nl"))

    dutch_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_mls_dutch(),
                                                    corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_dutch"),
                                                    lang="nl"))

    french_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_siwis_subset(),
                                                     corpus_dir=os.path.join(PREPROCESSING_DIR, "siwis"),
                                                     lang="fr"))

    french_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_blizzard2023_ad(),
                                                     corpus_dir=os.path.join(PREPROCESSING_DIR, "blizzard2023ad"),
                                                     lang="fr"))

    french_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_blizzard2023_neb(),
                                                     corpus_dir=os.path.join(PREPROCESSING_DIR, "blizzard2023neb"),
                                                     lang="fr"))

    french_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_mls_french(),
                                                     corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_french"),
                                                     lang="fr"))

    portuguese_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_mls_portuguese(),
                                                         corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_porto"),
                                                         lang="pt-br"))

    polish_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_mls_polish(),
                                                     corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_polish"),
                                                     lang="pl"))

    italian_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_mls_italian(),
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_italian"),
                                                      lang="it"))

    chinese_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_css10cmn(),
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_chinese"),
                                                      lang="cmn"))

    chinese_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_aishell3(),
                                                      corpus_dir=os.path.join(PREPROCESSING_DIR, "aishell3"),
                                                      lang="cmn"))

    vietnamese_datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_vietTTS(),
                                                         corpus_dir=os.path.join(PREPROCESSING_DIR, "vietTTS"),
                                                         lang="vi"))

    datasets.append(ConcatDataset(english_datasets))
    datasets.append(ConcatDataset(german_datasets))
    datasets.append(ConcatDataset(greek_datasets))
    datasets.append(ConcatDataset(spanish_datasets))
    datasets.append(ConcatDataset(finnish_datasets))
    datasets.append(ConcatDataset(russian_datasets))
    datasets.append(ConcatDataset(hungarian_datasets))
    datasets.append(ConcatDataset(dutch_datasets))
    datasets.append(ConcatDataset(french_datasets))
    datasets.append(ConcatDataset(portuguese_datasets))
    datasets.append(ConcatDataset(polish_datasets))
    datasets.append(ConcatDataset(italian_datasets))
    datasets.append(ConcatDataset(chinese_datasets))
    datasets.append(ConcatDataset(vietnamese_datasets))

    if generate_all_aligner_caches:
        return

    if remove_faulty_samples:
        find_and_remove_faulty_samples(net=PortaSpeech(),
                                       datasets=english_datasets +
                                                german_datasets +
                                                greek_datasets +
                                                spanish_datasets +
                                                finnish_datasets +
                                                russian_datasets +
                                                hungarian_datasets +
                                                dutch_datasets +
                                                french_datasets +
                                                portuguese_datasets +
                                                polish_datasets +
                                                italian_datasets +
                                                chinese_datasets +
                                                vietnamese_datasets,
                                       device=torch.device("cuda"),
                                       path_to_checkpoint=resume_checkpoint)
    model = PortaSpeech()
    if use_wandb:
        wandb.init(
            name=f"{__name__.split('.')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}" if wandb_resume_id is None else None,
            id=wandb_resume_id,  # this is None if not specified in the command line arguments.
            resume="must" if wandb_resume_id is not None else None)
    train_loop(net=model,
               device=torch.device("cuda"),
               datasets=datasets,
               batch_size=32,
               # choosing batchsize for this is tricky.
               # A large batchsize is dangerous for the VAE,
               # but a large batchsize is better for the modelling of the languages.
               # Starting from a single speaker single language model can help with this.
               save_directory=meta_save_dir,
               path_to_checkpoint=resume_checkpoint,
               path_to_embed_model=os.path.join(MODELS_DIR, "Embedding", "embedding_function.pt"),
               resume=resume,
               fine_tune=finetune,
               use_wandb=use_wandb)
    if use_wandb:
        wandb.finish()


@torch.inference_mode()
def find_and_remove_faulty_samples(net,
                                   datasets,
                                   device,
                                   path_to_checkpoint,
                                   path_to_embedding_checkpoint=os.path.join(MODELS_DIR, "Embedding",
                                                                             "embedding_function.pt")):
    net = net.to(device)
    torch.multiprocessing.set_sharing_strategy('file_system')
    check_dict = torch.load(os.path.join(path_to_checkpoint), map_location=device)
    net.load_state_dict(check_dict["model"])
    style_embedding_function = StyleEmbedding().to(device)
    check_dict = torch.load(path_to_embedding_checkpoint, map_location=device)
    style_embedding_function.load_state_dict(check_dict["style_emb_func"])
    for dataset_index in range(len(datasets)):
        nan_ids = list()
        for datapoint_index in tqdm(range(len(datasets[dataset_index]))):
            style_embedding = style_embedding_function(
                batch_of_spectrograms=datasets[dataset_index][datapoint_index][2].unsqueeze(0).to(device),
                batch_of_spectrogram_lengths=datasets[dataset_index][datapoint_index][3].to(device))
            loss = net(text_tensors=datasets[dataset_index][datapoint_index][0].unsqueeze(0).to(device),
                       text_lengths=datasets[dataset_index][datapoint_index][1].to(device),
                       gold_speech=datasets[dataset_index][datapoint_index][2].unsqueeze(0).to(device),
                       speech_lengths=datasets[dataset_index][datapoint_index][3].to(device),
                       gold_durations=datasets[dataset_index][datapoint_index][4].unsqueeze(0).to(device),
                       gold_pitch=datasets[dataset_index][datapoint_index][6].unsqueeze(0).to(device),
                       # mind the switched order
                       gold_energy=datasets[dataset_index][datapoint_index][5].unsqueeze(0).to(device),
                       # mind the switched order
                       utterance_embedding=style_embedding.unsqueeze(0).to(device),
                       lang_ids=datasets[dataset_index][datapoint_index][8].unsqueeze(0).to(device),
                       return_mels=False).squeeze()
            if torch.isnan(loss):
                print(f"NAN DETECTED: {dataset_index}, {datapoint_index}")
                nan_ids.append(datapoint_index)
        datasets[dataset_index].remove_samples(nan_ids)


def build_all_aligner_dataset_caches(device):
    factory = AlignerDatasetBuilder(device=device)
    factory.build_cache(transcript_dict=build_path_to_transcript_dict_nancy(),
                        corpus_dir=os.path.join(PREPROCESSING_DIR, "Nancy"),
                        lang="en")

    factory.build_cache(transcript_dict=build_path_to_transcript_dict_ljspeech(),
                        corpus_dir=os.path.join(PREPROCESSING_DIR, "LJSpeech"),
                        lang="en")

    factory.build_cache(transcript_dict=build_path_to_transcript_dict_libritts_all_clean(),
                        corpus_dir=os.path.join(PREPROCESSING_DIR, "libri_all_clean"),
                        lang="en")

    factory.build_cache(transcript_dict=build_path_to_transcript_dict_vctk(),
                        corpus_dir=os.path.join(PREPROCESSING_DIR, "vctk"),
                        lang="en")

    factory.build_cache(transcript_dict=build_path_to_transcript_dict_nvidia_hifitts(),
                        corpus_dir=os.path.join(PREPROCESSING_DIR, "hifi"),
                        lang="en")

    factory.build_cache(transcript_dict=build_path_to_transcript_dict_RAVDESS(),
                        corpus_dir=os.path.join(PREPROCESSING_DIR, "ravdess"),
                        lang="en")

    factory.build_cache(transcript_dict=build_path_to_transcript_dict_ESDS(),
                        corpus_dir=os.path.join(PREPROCESSING_DIR, "esds"),
                        lang="en")

    factory.build_cache(transcript_dict=build_path_to_transcript_dict_karlsson(),
                        corpus_dir=os.path.join(PREPROCESSING_DIR, "Karlsson"),
                        lang="de")

    factory.build_cache(transcript_dict=build_path_to_transcript_dict_eva(),
                        corpus_dir=os.path.join(PREPROCESSING_DIR, "Eva"),
                        lang="de")

    factory.build_cache(transcript_dict=build_path_to_transcript_dict_hokus(),
                        corpus_dir=os.path.join(PREPROCESSING_DIR, "Hokus"),
                        lang="de")

    factory.build_cache(transcript_dict=build_path_to_transcript_dict_bernd(),
                        corpus_dir=os.path.join(PREPROCESSING_DIR, "Bernd"),
                        lang="de")

    factory.build_cache(transcript_dict=build_path_to_transcript_dict_hui_others(),
                        corpus_dir=os.path.join(PREPROCESSING_DIR, "hui_others"),
                        lang="de")

    factory.build_cache(transcript_dict=build_path_to_transcript_dict_thorsten(),
                        corpus_dir=os.path.join(PREPROCESSING_DIR, "Thorsten"),
                        lang="de")

    factory.build_cache(transcript_dict=build_path_to_transcript_dict_css10el(),
                        corpus_dir=os.path.join(PREPROCESSING_DIR, "meta_Greek"),
                        lang="el")

    factory.build_cache(transcript_dict=build_path_to_transcript_dict_spanish_blizzard_train(),
                        corpus_dir=os.path.join(PREPROCESSING_DIR, "spanish_blizzard"),
                        lang="es")

    factory.build_cache(transcript_dict=build_path_to_transcript_dict_css10es(),
                        corpus_dir=os.path.join(PREPROCESSING_DIR, "meta_Spanish"),
                        lang="es")

    factory.build_cache(transcript_dict=build_path_to_transcript_dict_mls_spanish(),
                        corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_spanish"),
                        lang="es")

    factory.build_cache(transcript_dict=build_path_to_transcript_dict_css10fi(),
                        corpus_dir=os.path.join(PREPROCESSING_DIR, "meta_Finnish"),
                        lang="fi")

    factory.build_cache(transcript_dict=build_path_to_transcript_dict_css10ru(),
                        corpus_dir=os.path.join(PREPROCESSING_DIR, "meta_Russian"),
                        lang="ru")

    factory.build_cache(transcript_dict=build_path_to_transcript_dict_css10hu(),
                        corpus_dir=os.path.join(PREPROCESSING_DIR, "meta_Hungarian"),
                        lang="hu")

    factory.build_cache(transcript_dict=build_path_to_transcript_dict_css10nl(),
                        corpus_dir=os.path.join(PREPROCESSING_DIR, "meta_Dutch"),
                        lang="nl")

    factory.build_cache(transcript_dict=build_path_to_transcript_dict_mls_dutch(),
                        corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_dutch"),
                        lang="nl")

    factory.build_cache(transcript_dict=build_path_to_transcript_dict_css10fr(),
                        corpus_dir=os.path.join(PREPROCESSING_DIR, "meta_French"),
                        lang="fr")

    factory.build_cache(transcript_dict=build_path_to_transcript_dict_mls_french(),
                        corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_french"),
                        lang="fr")

    factory.build_cache(transcript_dict=build_path_to_transcript_dict_mls_portuguese(),
                        corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_porto"),
                        lang="pt-br")

    factory.build_cache(transcript_dict=build_path_to_transcript_dict_mls_polish(),
                        corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_polish"),
                        lang="pl")

    factory.build_cache(transcript_dict=build_path_to_transcript_dict_mls_italian(),
                        corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_italian"),
                        lang="it")

    factory.build_cache(transcript_dict=build_path_to_transcript_dict_css10cmn(),
                        corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_chinese"),
                        lang="cmn")

    factory.build_cache(transcript_dict=build_path_to_transcript_dict_aishell3(),
                        corpus_dir=os.path.join(PREPROCESSING_DIR, "aishell3"),
                        lang="cmn")

    factory.build_cache(transcript_dict=build_path_to_transcript_dict_vietTTS(),
                        corpus_dir=os.path.join(PREPROCESSING_DIR, "vietTTS"),
                        lang="vi")
