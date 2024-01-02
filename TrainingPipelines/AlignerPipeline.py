from torch.utils.data import ConcatDataset

from Architectures.Aligner.autoaligner_train_loop import train_loop as train_aligner
from Utility.corpus_preparation import prepare_aligner_corpus
from Utility.path_to_transcript_dicts import *
from Utility.storage_config import MODELS_DIR
from Utility.storage_config import PREPROCESSING_DIR


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume, use_wandb, wandb_resume_id, gpu_count):
    if gpu_id == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    if gpu_count > 1:
        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(backend="nccl")
    else:
        rank = 0

    print("Preparing")

    datasets = list()

    lang_id = "afr"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_nchlt_afr(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "nchlt_afr"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "nso"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_nchlt_nso(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "nchlt_nso"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "sot"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_nchlt_sot(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "nchlt_sot"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "ssw"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_nchlt_ssw(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "nchlt_ssw"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "tsn"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_nchlt_tsn(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "nchlt_tsn"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "tso"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_nchlt_tso(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "nchlt_tso"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "ven"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_nchlt_ven(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "nchlt_ven"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "xho"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_nchlt_xho(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "nchlt_xho"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "zul"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_nchlt_zul(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "nchlt_zul"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "bem"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_bembaspeech(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "bembaspeech"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "swh"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_alffa_sw(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "alffa_sw"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "amh"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_alffa_am(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "alffa_am"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "wol"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_alffa_wo(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "alffa_wo"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "mal"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_malayalam(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "malayalam"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "mal"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_msc(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "msc"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "chv"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_chuvash(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "chuvash"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "iba"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_iban(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "iban"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "sun"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_sundanese_speech(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "sundanese_speech"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "sin"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_sinhala_speech(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "sinhala_speech"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "ben"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_bengali_speech(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "bengali_speech"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "npi"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_nepali_speech(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "nepali_speech"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "jav"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_javanese_speech(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "javanese_speech"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "fon"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_african_voices_fon_alf(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_fon_alf"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "hau"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_african_voices_hausa_cmv(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_hausa_cmv"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "lbb"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_african_voices_ibibio_lst(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_ibibio_lst"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "kik"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_african_voices_kikuyu_opb(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_kikuyu_opb"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "lin"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_african_voices_lingala_opb(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_lingala_opb"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "lug"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_african_voices_ganda_cmv(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_ganda_cmv"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "luo"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_african_voices_luo_afv(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_luo_afv"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "luo"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_african_voices_luo_opb(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_luo_opb"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "swh"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_african_voices_swahili_llsti(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_swahili_llsti"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "sxb"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_african_voices_suba_afv(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_suba_afv"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "wol"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_african_voices_wolof_alf(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_wolof_alf"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "yor"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_african_voices_yoruba_opb(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "african_voices_yoruba_opb"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "nya"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_zambezi_voice_nyanja(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "zambezi_voice_nyanja"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "loz"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_zambezi_voice_lozi(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "zambezi_voice_lozi"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "toi"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_zambezi_voice_tonga(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "zambezi_voice_tonga"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "afr"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_afrikaans(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_afrikaans"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "amh"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_amharic(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_amharic"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "arb"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_arabic(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_arabic"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "asm"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_assamese(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_assamese"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "ast"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_asturian(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_asturian"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "azj"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_azerbaijani(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_azerbaijani"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "bel"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_belarusian(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_belarusian"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "bul"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_bulgarian(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_bulgarian"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "ben"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_bengali(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_bengali"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "bos"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_bosnian(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_bosnian"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "cat"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_catalan(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_catalan"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "ceb"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_cebuano(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_cebuano"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "sdh"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_sorani_kurdish(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_sorani_kurdish"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "cmn"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_mandarin(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_mandarin"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "ces"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_czech(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_czech"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "cym"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_welsh(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_welsh"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "dan"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_danish(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_danish"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "deu"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_german(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_german"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "ell"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_greek(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_greek"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "eng"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_english(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_english"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "spa"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_spanish(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_spanish"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "ekk"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_estonian(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_estonian"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "pes"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_persian(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_persian"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "ful"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_fula(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_fula"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "fin"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_finnish(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_finnish"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "fil"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_filipino(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_filipino"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "fra"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_french(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_french"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "gle"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_irish(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_irish"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "glg"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_galician(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_galician"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "guj"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_gujarati(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_gujarati"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "hau"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_hausa(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_hausa"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "heb"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_hebrew(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_hebrew"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "hin"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_hindi(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_hindi"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "hrv"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_croatian(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_croatian"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "hun"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_hungarian(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_hungarian"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "hye"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_armenian(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_armenian"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "ind"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_indonesian(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_indonesian"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "ibo"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_igbo(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_igbo"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "isl"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_icelandic(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_icelandic"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "ita"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_italian(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_italian"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    # lang_id = "jpn"
    # datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_japanese(),
    #                                       corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_japanese"),
    #                                       lang=lang_id,
    #                                       gpu_count=gpu_count,
    #                                       rank=rank, device=device))
    lang_id = "jav"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_javanese(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_javanese"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "kat"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_georgian(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_georgian"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "kam"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_kamba(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_kamba"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "kea"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_kabuverdianu(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_kabuverdianu"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "kaz"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_kazakh(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_kazakh"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "khm"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_khmer(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_khmer"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "kan"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_kannada(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_kannada"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "kor"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_korean(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_korean"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    # lang_id = "kir"
    # datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_kyrgyz(),
    #                                       corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_kyrgyz"),
    #                                       lang=lang_id,
    #                                       gpu_count=gpu_count,
    #                                       rank=rank, device=device))
    lang_id = "ltz"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_luxembourgish(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_luxembourgish"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "lug"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_ganda(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_ganda"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "lin"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_lingala(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_lingala"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "lao"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_lao(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_lao"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "lit"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_lithuanian(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_lithuanian"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "luo"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_luo(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_luo"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "lvs"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_latvian(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_latvian"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "mri"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_maori(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_maori"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "mkd"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_macedonian(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_macedonian"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "mal"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_malayalam(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_malayalam"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "xng"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_mongolian(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_mongolian"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "mar"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_marathi(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_marathi"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "zsm"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_malay(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_malay"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "mlt"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_maltese(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_maltese"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    # lang_id = "mya"
    # datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_burmese(),
    #                                       corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_burmese"),
    #                                       lang=lang_id,
    #                                       gpu_count=gpu_count,
    #                                       rank=rank, device=device))
    # lang_id = "nob"
    # datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_norwegian(),
    #                                       corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_norwegian"),
    #                                       lang=lang_id,
    #                                       gpu_count=gpu_count,
    #                                       rank=rank, device=device))
    # lang_id = "npi"
    # datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_nepali(),
    #                                       corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_nepali"),
    #                                       lang=lang_id,
    #                                       gpu_count=gpu_count,
    #                                       rank=rank, device=device))
    lang_id = "nld"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_dutch(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_dutch"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "sot"  # technically incorrect, this is the shorthand for southern sotho, but it seems northerns sotho is not in out list.
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_northern_sotho(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_northern_sotho"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "nya"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_nyanja(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_nyanja"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "oci"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_occitan(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_occitan"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "orm"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_oroma(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_oroma"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "ory"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_oriya(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_oriya"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "pan"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_punjabi(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_punjabi"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "pol"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_polish(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_polish"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "pst"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_pashto(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_pashto"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "por"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_portuguese(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_portuguese"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "ron"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_romanian(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_romanian"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "rus"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_russian(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_russian"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "snd"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_sindhi(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_sindhi"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "slk"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_slovak(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_slovak"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "slv"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_slovenian(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_slovenian"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "sna"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_shona(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_shona"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "som"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_somali(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_somali"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "srp"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_serbian(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_serbian"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "swe"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_swedish(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_swedish"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "swh"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_swahili(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_swahili"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "tam"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_tamil(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_tamil"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "tel"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_telugu(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_telugu"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "tgk"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_tajik(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_tajik"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    # lang_id = "tha"
    # datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_thai(),
    #                                       corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_thai"),
    #                                       lang=lang_id,
    #                                       gpu_count=gpu_count,
    #                                       rank=rank, device=device))
    lang_id = "tur"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_turkish(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_turkish"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "urk"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_ukrainian(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_ukrainian"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "umb"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_umbundu(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_umbundu"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "urd"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_urdu(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_urdu"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "uzn"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_uzbek(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_uzbek"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "vie"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_vietnamese(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_vietnamese"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "wol"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_wolof(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_wolof"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "xho"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_xhosa(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_xhosa"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "yor"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_yoruba(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_yoruba"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    # lang_id = "yue"
    # datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_cantonese(),
    #                                       corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_cantonese"),
    #                                       lang=lang_id,
    #                                       gpu_count=gpu_count,
    #                                       rank=rank, device=device))
    lang_id = "zul"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_fleurs_zulu(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "fleurs_zulu"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "gle"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_living_audio_dataset_irish(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "living_audio_dataset_irish"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "nld"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_living_audio_dataset_dutch(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "living_audio_dataset_dutch"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "rus"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_living_audio_dataset_russian(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "living_audio_dataset_russian"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "ron"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_romanian_db(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "romanian_db"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "fas"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_shemo(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "shemo"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "eng"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_mslt_english(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "mslt_english"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "jap"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_mslt_japanese(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "mslt_japanese"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "cmn"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_mslt_chinese(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "mslt_chinese"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    # lang_id = "hin"
    # datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_rajasthani_hindi_speech(),
    #                                       corpus_dir=os.path.join(PREPROCESSING_DIR, "rajasthani_hindi_speech"),
    #                                       lang=lang_id,
    #                                       gpu_count=gpu_count,
    #                                       rank=rank, device=device))
    lang_id = "eng"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_cmu_arctic(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "cmu_arctic"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    # lang_id = "tat"
    # datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_sevil_tatar(),
    #                                       corpus_dir=os.path.join(PREPROCESSING_DIR, "sevil_tatar"),
    #                                       lang=lang_id,
    #                                       gpu_count=gpu_count,
    #                                       rank=rank, device=device))
    lang_id = "ara"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_clartts(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "clartts"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "bhd"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_snow_mountain_bhadrawahi(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_bhadrawahi"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "kfs"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_snow_mountain_bilaspuri(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_bilaspuri"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "dgo"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_snow_mountain_dogri(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_dogri"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "gbk"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_snow_mountain_gaddi(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_gaddi"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "bgc"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_snow_mountain_haryanvi(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_haryanvi"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "hin"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_snow_mountain_hindi(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_hindi"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "xnr"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_snow_mountain_kangri(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_kangri"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "kan"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_snow_mountain_kannada(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_kannada"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "kfx"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_snow_mountain_kulvi(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_kulvi"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "kfx"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_snow_mountain_kulvi_outer_seraji(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_kulvi_outer_seraji"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "mal"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_snow_mountain_malayalam(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_malayalam"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "mjl"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_snow_mountain_mandeali(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_mandeali"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "bfz"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_snow_mountain_pahari_mahasui(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_pahari_mahasui"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "tam"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_snow_mountain_tamil(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_tamil"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "tel"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_snow_mountain_telugu(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "snow_mountain_telugu"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "ukr"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_ukrainian_lada(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "ukrainian_lada"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "deu"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_m_ailabs_german(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "m_ailabs_german"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "spa"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_m_ailabs_spanish(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "m_ailabs_spanish"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "fra"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_m_ailabs_french(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "m_ailabs_french"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "ita"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_m_ailabs_italian(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "m_ailabs_italian"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "pol"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_m_ailabs_polish(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "m_ailabs_polish"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "rus"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_m_ailabs_russian(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "m_ailabs_russian"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))
    lang_id = "ukr"
    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_m_ailabs_ukrainian(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "m_ailabs_ukrainian"),
                                           lang=lang_id,
                                           gpu_count=gpu_count,
                                           rank=rank, device=device))

    for lang in ["acf", "bss", "deu", "inb", "nca", "quh", "wap", "acr", "bus", "dgr", "ind", "maz", "nch", "qul", "tav", "wmw", "acu", "byr", "dik", "iou", "mbb", "ncj", "qvc", "tbc", "xed", "agd", "bzh", "djk", "ipi", "mbc", "ncl", "qve", "tbg", "xon", "agg", "bzj", "dop", "jac", "mbh", "ncu", "qvh", "tbl", "xtd", "agn",
                 "caa", "jic", "mbj", "ndj", "qvm", "tbz", "xtm", "agr", "cab", "emp", "jiv", "mbt", "nfa", "qvn", "tca", "yaa", "agu", "cap", "eng", "jvn", "mca", "ngp", "qvs", "tcs", "yad", "aia", "car", "ese", "mcb", "ngu", "qvw", "yal", "cax", "kaq", "mcd", "nhe", "qvz", "tee", "ycn", "ake", "cbc",
                 "far", "mco", "qwh", "yka", "alp", "cbi", "fra", "kdc", "mcp", "nhu", "qxh", "ter", "ame", "cbr", "gai", "kde", "mcq", "nhw", "qxn", "tew", "yre", "amf", "cbs", "gam", "kdl", "mdy", "nhy", "qxo", "tfr", "yva", "amk", "cbt", "geb", "kek", "med", "nin", "rai", "tgk", "zaa", "apb", "cbu", "glk",
                 "ken", "mee", "nko", "rgu", "zab", "apr", "cbv", "meq", "nld", "tgo", "zac", "arl", "cco", "gng", "kje", "met", "nlg", "rop", "tgp", "zad", "grc", "klv", "mgh", "nnq", "rro", "zai", "ata", "cek", "gub", "kmu", "mib", "noa", "ruf", "tna", "zam", "atb", "cgc", "guh", "kne",
                 "mie", "not", "rug", "tnk", "zao", "atg", "chf", "knf", "mih", "npl", "rus", "tnn", "zar", "awb", "chz", "gum", "knj", "mil", "sab", "tnp", "zas", "cjo", "guo", "ksr", "mio", "obo", "seh", "toc", "zav", "azg", "cle", "gux", "kue", "mit", "omw", "sey", "tos", "zaw", "azz", "cme", "gvc", "kvn", "miz",
                 "ood", "sgb", "tpi", "zca", "bao", "cni", "gwi", "kwd", "mkl", "shp", "tpt", "zga", "bba", "cnl", "gym", "kwf", "mkn", "ote", "sja", "trc", "ziw", "bbb", "cnt", "gyr", "kwi", "mop", "otq", "snn", "ttc", "zlm", "cof", "hat", "kyc", "mox", "pab", "snp", "tte", "zos", "bgt", "con", "kyf", "mpm", "pad",
                 "som", "tue", "zpc", "bjr", "cot", "heb", "kyg", "mpp", "soy", "tuf", "zpl", "bjv", "cpa", "kyq", "mpx", "pao", "spa", "tuo", "zpm", "bjz", "cpb", "hlt", "kyz", "mqb", "pib", "spp", "tur", "zpo", "bkd", "cpu", "hns", "lac", "mqj", "pir", "spy", "txq", "zpu", "blz", "crn", "hto", "lat", "msy", "pjt", "sri",
                 "txu", "zpz", "bmr", "cso", "hub", "lex", "mto", "pls", "srm", "udu", "ztq", "bmu", "ctu", "hui", "lgl", "muy", "poi", "srn", "ukr", "zty", "bnp", "cuc", "lid", "mxb", "pol", "stp", "upv", "zyp", "boa", "cui", "huu", "mxq", "por", "sus", "ura", "boj", "cuk", "huv", "llg", "mxt", "poy", "suz", "urb", "box",
                 "cwe", "hvn", "prf", "swe", "urt", "bpr", "cya", "ign", "lww", "myk", "ptu", "swh", "usp", "bps", "daa", "ikk", "maj", "myy", "pwg", "sxb", "vid", "bqc", "dah", "nab", "qub", "tac", "vie", "bqp", "ded", "imo", "maq", "nas", "quf", "taj", "vmy"]:
        datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_mms_template(lang=lang),
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, f"mms_{lang}"),
                                               lang=f"{lang}",
                                               gpu_count=gpu_count,
                                               rank=rank, device=device))

    # ENGLISH

    chunk_count = 50
    chunks = split_dictionary_into_chunks(build_path_to_transcript_dict_mls_english(), split_n=chunk_count)
    for index in range(chunk_count):
        datasets.append(prepare_aligner_corpus(transcript_dict=chunks[index],
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, f"mls_english_chunk_{index}"),
                                               lang="eng",
                                               device=device,
                                               gpu_count=gpu_count,
                                               rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_nancy,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "Nancy"),
                                           lang="eng",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_ryanspeech,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "Ryan"),
                                           lang="eng",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_ljspeech,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "LJSpeech"),
                                           lang="eng",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_libritts_all_clean,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "libri_all_clean"),
                                           lang="eng",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_vctk,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "vctk"),
                                           lang="eng",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_nvidia_hifitts,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "hifi"),
                                           lang="eng",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_CREMA_D,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "cremad"),
                                           lang="eng",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_EmoV_DB,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "emovdb"),
                                           lang="eng",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_RAVDESS,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "ravdess"),
                                           lang="eng",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_ESDS,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "esds"),
                                           lang="eng",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_blizzard_2013,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "blizzard2013"),
                                           lang="eng",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_jenny,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "jenny"),
                                           lang="eng",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    # GERMAN

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_karlsson,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "Karlsson"),
                                           lang="deu",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_eva,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "Eva"),
                                           lang="deu",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_hokus,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "Hokus"),
                                           lang="deu",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_bernd,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "Bernd"),
                                           lang="deu",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_friedrich,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "Friedrich"),
                                           lang="deu",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_hui_others,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "hui_others"),
                                           lang="deu",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_thorsten_emotional(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "thorsten_emotional"),
                                           lang="deu",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_thorsten_neutral(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "thorsten_neutral"),
                                           lang="deu",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_thorsten_2022_10(),
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "thorsten_2022"),
                                           lang="deu",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    chunk_count = 10
    chunks = split_dictionary_into_chunks(build_path_to_transcript_dict_mls_german(), split_n=chunk_count)
    for index in range(chunk_count):
        datasets.append(prepare_aligner_corpus(transcript_dict=chunks[index],
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, f"mls_german_chunk_{index}"),
                                               lang="deu",
                                               device=device,
                                               gpu_count=gpu_count,
                                               rank=rank))

    # FRENCH

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10fr,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_French"),
                                           lang="fra",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_mls_french,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_french"),
                                           lang="fra",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_blizzard2023_ad_silence_removed,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "ad_e"),
                                           lang="fra",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_blizzard2023_neb_silence_removed,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "neb"),
                                           lang="fra",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_blizzard2023_neb_e_silence_removed,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "neb_e"),
                                           lang="fra",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_synpaflex_norm_subset,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "synpaflex"),
                                           lang="fra",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_siwis_subset,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "siwis"),
                                           lang="fra",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    # SPANISH

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_mls_spanish,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_spanish"),
                                           lang="spa",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10es,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Spanish"),
                                           lang="spa",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_spanish_blizzard_train,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "spanish_blizzard"),
                                           lang="spa",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    # CHINESE

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10cmn,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_chinese"),
                                           lang="cmn",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_aishell3,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "aishell3"),
                                           lang="cmn",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    # PORTUGUESE

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_mls_portuguese,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_porto"),
                                           lang="por",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    # POLISH

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_mls_polish,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_polish"),
                                           lang="pol",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    # ITALIAN

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_mls_italian,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_italian"),
                                           lang="ita",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    # DUTCH

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_mls_dutch,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "mls_dutch"),
                                           lang="nld",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10nl,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Dutch"),
                                           lang="nld",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    # GREEK

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10el,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Greek"),
                                           lang="ell",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    # FINNISH

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10fi,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Finnish"),
                                           lang="fin",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    # VIETNAMESE

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_VIVOS_viet,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "VIVOS_viet"),
                                           lang="vie",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    # RUSSIAN

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10ru,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Russian"),
                                           lang="rus",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    # HUNGARIAN

    datasets.append(prepare_aligner_corpus(transcript_dict=build_path_to_transcript_dict_css10hu,
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, "css10_Hungarian"),
                                           lang="hun",
                                           device=device,
                                           gpu_count=gpu_count,
                                           rank=rank))

    train_set = ConcatDataset(datasets)
    save_dir = os.path.join(MODELS_DIR, "Aligner")
    os.makedirs(save_dir, exist_ok=True)
    save_dir_aligner = save_dir + "/aligner"
    os.makedirs(save_dir_aligner, exist_ok=True)

    train_aligner(train_dataset=train_set,
                  device=device,
                  save_directory=save_dir,
                  steps=1500000,
                  batch_size=16,
                  path_to_checkpoint=resume_checkpoint,
                  fine_tune=finetune,
                  debug_img_path=save_dir_aligner,
                  resume=resume,
                  gpu_count=gpu_count,
                  rank=rank,
                  steps_per_checkpoint=5000)
