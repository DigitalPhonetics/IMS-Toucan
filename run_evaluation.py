import os
import argparse

import torch
from transformers import pipeline
from speechbrain.pretrained import EncoderClassifier

from Utility.storage_config import PREPROCESSING_DIR
from Evaluation.objective_evaluation import *
from Preprocessing.sentence_embeddings.EmotionRoBERTaSentenceEmbeddingExtractor import EmotionRoBERTaSentenceEmbeddingExtractor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--gpu_id',
                        type=str,
                        help="Which GPU to run on. If not specified runs on CPU.",
                        default="cpu")
    args = parser.parse_args()
    if args.gpu_id == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")
        print(f"No GPU specified, using CPU.")

    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"
        device = torch.device("cuda")
        print(f"Making GPU {os.environ['CUDA_VISIBLE_DEVICES']} the only visible device.")

    num_test_sentences = 50 # 50 sentences per emotion category

    # extract test sentences
    # test sentences are a dict with shape {dataset: {emotion: [sentences]}}
    print("Loading test senteces...")
    if not os.path.exists(os.path.join(PREPROCESSING_DIR, "Evaluation", "test_sentences.pt")):
        os.makedirs(os.path.join(PREPROCESSING_DIR, "Evaluation"), exist_ok=True)

        emotion_to_sents_dialog = extract_dailydialogue_sentences()
        
        tales_data_dir = "/mount/arbeitsdaten/synthesis/bottts/IMS-Toucan/Corpora/Tales"
        emotion_to_sents_tales = extract_tales_sentences(tales_data_dir)
    
        classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1, device=device)
        test_sentences = {}
        test_sentences["dailydialogues"] = get_sorted_test_sentences(emotion_to_sents_dialog, classifier)
        test_sentences["tales"] = get_sorted_test_sentences(emotion_to_sents_tales, classifier)

        torch.save(test_sentences, os.path.join(PREPROCESSING_DIR, "Evaluation", "test_sentences.pt"))
    else:
        test_sentences = torch.load(os.path.join(PREPROCESSING_DIR, "Evaluation", "test_sentences.pt"), map_location='cpu')

    for dataset, emotion_to_sents in test_sentences.items():
        for emotion, sentences in emotion_to_sents.items():
            test_sentences[dataset][emotion] = sentences[:num_test_sentences]
    
    for dataset, emotion_to_sents in test_sentences.items():
        for emotion, sentences in emotion_to_sents.items():
            if len(sentences) != num_test_sentences:
                raise ValueError(f"Number of sentences is not {num_test_sentences} for dataset {dataset} and emotion {emotion}.")

    # synthesize test sentences
    if not os.path.exists(os.path.join("./audios/Evaluation")):
        print("Synthesizing Baseline...")
        synthesize_test_sentences(version="Baseline",
                                exec_device=device,
                                biggan=False,
                                sent_emb_extractor=None,
                                test_sentences=test_sentences,
                                silent=True)
        print("Synthesizing Proposed...")
        sent_emb_extractor = EmotionRoBERTaSentenceEmbeddingExtractor(pooling="cls")
        synthesize_test_sentences(version="Sent",
                                exec_device=device,
                                biggan=False,
                                sent_emb_extractor=sent_emb_extractor,
                                test_sentences=test_sentences,
                                silent=True)
    
    # extract speaker embeddings
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", 
                                                savedir="./Models/Embedding/spkrec-xvect-voxceleb", 
                                                run_opts={"device": device})
    speaker_embeddings = extract_speaker_embeddings("./audios/Evaluation", classifier)
