import os
import argparse
from statistics import median, mean

import torch
from transformers import pipeline
from speechbrain.pretrained import EncoderClassifier
from speechbrain.pretrained.interfaces import foreign_class
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from Utility.storage_config import PREPROCESSING_DIR, MODELS_DIR
from Evaluation.objective_evaluation import *
from Preprocessing.sentence_embeddings.EmotionRoBERTaSentenceEmbeddingExtractor import EmotionRoBERTaSentenceEmbeddingExtractor
from InferenceInterfaces.InferenceArchitectures.InferenceAvocodo import HiFiGANGenerator

import sys

NUM_TEST_SENTENCES = 50 # 50 sentences per emotion category
EMOTIONS = ["anger", "joy", "neutral", "sadness", "surprise"]

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
            test_sentences[dataset][emotion] = sentences[:NUM_TEST_SENTENCES]
    
    for dataset, emotion_to_sents in test_sentences.items():
        for emotion, sentences in emotion_to_sents.items():
            if len(sentences) != NUM_TEST_SENTENCES:
                raise ValueError(f"Number of sentences is not {NUM_TEST_SENTENCES} for dataset {dataset} and emotion {emotion}.")

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
        print("Synthesizing Prompt...")
        synthesize_test_sentences(version="Prompt",
                                exec_device=device,
                                biggan=False,
                                sent_emb_extractor=sent_emb_extractor,
                                test_sentences=test_sentences,
                                silent=True)
    
    # get vocoded original sentences
    if not os.path.exists(os.path.join("./audios/Evaluation/Original")):
        print("Vocoding Original...")
        os.makedirs(os.path.join("./audios/Evaluation/Original"), exist_ok=True)
        vocoder_model_path = os.path.join(MODELS_DIR, "Avocodo", "best.pt")
        mel2wav = HiFiGANGenerator(path_to_weights=vocoder_model_path).to(device)
        mel2wav.remove_weight_norm()
        mel2wav.eval()
        vocode_original(mel2wav, num_sentences=NUM_TEST_SENTENCES, device=device)
    
    # extract speaker embeddings
    if not os.path.exists(os.path.join(PREPROCESSING_DIR, "Evaluation", "speaker_embeddings_original.pt")):
        print("Extracting speaker embeddings...")
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", 
                                                    savedir="./Models/Embedding/spkrec-xvect-voxceleb", 
                                                    run_opts={"device": device})
        # shape {speaker: {emotion: {file_id: embedding}}}
        speaker_embeddings_original = extract_speaker_embeddings("./audios/Evaluation", classifier, version='Original')
        torch.save(speaker_embeddings_original, os.path.join(PREPROCESSING_DIR, "Evaluation", "speaker_embeddings_original.pt"))
        # shape {dataset: {speaker: {emotion: {file_id: embedding}}}}
        speaker_embeddings_baseline = extract_speaker_embeddings("./audios/Evaluation", classifier, version='Baseline')
        torch.save(speaker_embeddings_baseline, os.path.join(PREPROCESSING_DIR, "Evaluation", "speaker_embeddings_baseline.pt"))
        # shape {dataset: {speaker: {emotion: {file_id: embedding}}}}
        speaker_embeddings_sent = extract_speaker_embeddings("./audios/Evaluation", classifier, version='Sent')
        torch.save(speaker_embeddings_sent, os.path.join(PREPROCESSING_DIR, "Evaluation", "speaker_embeddings_sent.pt"))
        # shape {dataset: {speaker: {emotion: {prompt_emotion: {file_id: embedding}}}}}
        speaker_embeddings_prompt = extract_speaker_embeddings("./audios/Evaluation", classifier, version='Prompt')
        torch.save(speaker_embeddings_prompt, os.path.join(PREPROCESSING_DIR, "Evaluation", "speaker_embeddings_prompt.pt"))
    else:
        speaker_embeddings_original = torch.load(os.path.join(PREPROCESSING_DIR, "Evaluation", "speaker_embeddings_original.pt"), map_location='cpu')
        speaker_embeddings_baseline = torch.load(os.path.join(PREPROCESSING_DIR, "Evaluation", "speaker_embeddings_baseline.pt"), map_location='cpu')
        speaker_embeddings_sent = torch.load(os.path.join(PREPROCESSING_DIR, "Evaluation", "speaker_embeddings_sent.pt"), map_location='cpu')
        speaker_embeddings_prompt = torch.load(os.path.join(PREPROCESSING_DIR, "Evaluation", "speaker_embeddings_prompt.pt"), map_location='cpu')

    # calculate speaker similarity
    if not os.path.exists(os.path.join(PREPROCESSING_DIR, "Evaluation", "speaker_similarities_baseline.pt")):
        print("Calculating speaker similarity...")
        # shape {dataset: {speaker: {emotion: speaker_similarity}}}
        speaker_similarities_baseline = compute_speaker_similarity(speaker_embeddings_original, speaker_embeddings_baseline, version='Baseline')
        # shape {dataset: {speaker: {emotion: speaker_similarity}}}
        speaker_similarities_sent = compute_speaker_similarity(speaker_embeddings_original, speaker_embeddings_sent, version='Sent')
        # shape {dataset: {speaker: {prompt_emotion: speaker_similarity}}}
        speaker_similarities_prompt = compute_speaker_similarity(speaker_embeddings_original, speaker_embeddings_prompt, version='Prompt')

        torch.save(speaker_similarities_baseline, os.path.join(PREPROCESSING_DIR, "Evaluation", "speaker_similarities_baseline.pt"))
        torch.save(speaker_similarities_sent, os.path.join(PREPROCESSING_DIR, "Evaluation", "speaker_similarities_sent.pt"))
        torch.save(speaker_similarities_prompt, os.path.join(PREPROCESSING_DIR, "Evaluation", "speaker_similarities_prompt.pt"))

    # calculate word error rate
    print("Calculating word error rate...")
    if not os.path.exists(os.path.join(PREPROCESSING_DIR, "Evaluation", "transcriptions_original.pt")):
        print("Transcribing...")
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", cache_dir=os.path.join(MODELS_DIR, "ASR"))
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", cache_dir=os.path.join(MODELS_DIR, "ASR")).to(device)
        # shape {speaker: {emotion: {sentence_id: transcription}}}
        transcriptions_original = asr_transcribe("./audios/Evaluation", processor, model, version='Original')
        torch.save(transcriptions_original, os.path.join(PREPROCESSING_DIR, "Evaluation", "transcriptions_original.pt"))
        # shape {dataset: {speaker: {emotion: {sentence_id: transcription}}}}
        transcriptions_baseline = asr_transcribe("./audios/Evaluation", processor, model, version='Baseline')
        torch.save(transcriptions_baseline, os.path.join(PREPROCESSING_DIR, "Evaluation", "transcriptions_baseline.pt"))
        # shape {dataset: {speaker: {emotion: {sentence_id: transcription}}}}
        transcriptions_sent = asr_transcribe("./audios/Evaluation", processor, model, version='Sent')
        torch.save(transcriptions_sent, os.path.join(PREPROCESSING_DIR, "Evaluation", "transcriptions_sent.pt"))
        # shape {dataset: {speaker: {emotion: {prompt_emotion: {file_id: embedding}}}}}
        transcriptions_prompt = asr_transcribe("./audios/Evaluation", processor, model, version='Prompt')
        torch.save(transcriptions_prompt, os.path.join(PREPROCESSING_DIR, "Evaluation", "transcriptions_prompt.pt"))
    else:
        transcriptions_original = torch.load(os.path.join(PREPROCESSING_DIR, "Evaluation", "transcriptions_original.pt"), map_location='cpu')
        transcriptions_baseline = torch.load(os.path.join(PREPROCESSING_DIR, "Evaluation", "transcriptions_baseline.pt"), map_location='cpu')
        transcriptions_sent = torch.load(os.path.join(PREPROCESSING_DIR, "Evaluation", "transcriptions_sent.pt"), map_location='cpu')
        transcriptions_prompt = torch.load(os.path.join(PREPROCESSING_DIR, "Evaluation", "transcriptions_prompt.pt"), map_location='cpu')

    if not os.path.exists(os.path.join(PREPROCESSING_DIR, "Evaluation", "wers_original.pt")):
        # shape {speaker: {emotion: wer}}
        wers_original = compute_word_error_rate(transcriptions_original, test_sentences, version='Original')
        # shape {dataset: {speaker: {emotion: wer}}}
        wers_baseline = compute_word_error_rate(transcriptions_baseline, test_sentences, version='Baseline')
        # shape {dataset: {speaker: {emotion: wer}}}
        wers_sent = compute_word_error_rate(transcriptions_sent, test_sentences, version='Sent')
        # shape {dataset: {speaker: {prompt_emotion: wer}}}
        wers_prompt = compute_word_error_rate(transcriptions_prompt, test_sentences, version='Prompt')

        torch.save(wers_original, os.path.join(PREPROCESSING_DIR, "Evaluation", "wers_original.pt"))
        torch.save(wers_baseline, os.path.join(PREPROCESSING_DIR, "Evaluation", "wers_baseline.pt"))
        torch.save(wers_sent, os.path.join(PREPROCESSING_DIR, "Evaluation", "wers_sent.pt"))
        torch.save(wers_prompt, os.path.join(PREPROCESSING_DIR, "Evaluation", "wers_prompt.pt"))

    # speech emotion recognition
    print("Calculating speech emotion recognition...")
    if not os.path.exists(os.path.join(PREPROCESSING_DIR, "Evaluation", "predicted_emotions_original.pt")):
        print("Speech emotion recognition...")
        classifier = foreign_class(source=os.path.join(MODELS_DIR, "Emotion_Recognition"), 
                                   pymodule_file="custom_interface.py",
                                   classname="CustomEncoderWav2vec2Classifier",
                                   savedir=os.path.join(MODELS_DIR, "Emotion_Recognition"),
                                   run_opts={"device":device})
        
        # shape {speaker: {emotion: {sentence_id: predicted emotion}}}
        predicted_emotions_original = classify_speech_emotion("./audios/Evaluation", classifier, version='Original')
        torch.save(predicted_emotions_original, os.path.join(PREPROCESSING_DIR, "Evaluation", "predicted_emotions_original.pt"))
        # shape {dataset: {speaker: {emotion: {sentence_id: predicted emotion}}}}
        predicted_emotions_baseline = classify_speech_emotion("./audios/Evaluation", classifier, version='Baseline')
        torch.save(predicted_emotions_baseline, os.path.join(PREPROCESSING_DIR, "Evaluation", "predicted_emotions_baseline.pt"))
        # shape {dataset: {speaker: {emotion: {sentence_id: predicted emotion}}}}
        predicted_emotions_sent = classify_speech_emotion("./audios/Evaluation", classifier, version='Sent')
        torch.save(predicted_emotions_sent, os.path.join(PREPROCESSING_DIR, "Evaluation", "predicted_emotions_sent.pt"))
        # shape {dataset: {speaker: {emotion: {prompt_emotion: {sentence_id: predicted emotion}}}}}
        predicted_emotions_prompt = classify_speech_emotion("./audios/Evaluation", classifier, version='Prompt')
        torch.save(predicted_emotions_prompt, os.path.join(PREPROCESSING_DIR, "Evaluation", "predicted_emotions_prompt.pt"))
    else:
        predicted_emotions_original = torch.load(os.path.join(PREPROCESSING_DIR, "Evaluation", "predicted_emotions_original.pt"), map_location='cpu')
        predicted_emotions_baseline = torch.load(os.path.join(PREPROCESSING_DIR, "Evaluation", "predicted_emotions_baseline.pt"), map_location='cpu')
        predicted_emotions_sent = torch.load(os.path.join(PREPROCESSING_DIR, "Evaluation", "predicted_emotions_sent.pt"), map_location='cpu')
        predicted_emotions_prompt = torch.load(os.path.join(PREPROCESSING_DIR, "Evaluation", "predicted_emotions_prompt.pt"), map_location='cpu')

    # shape {speaker: {emotion: {pred_emotion: count}}}
    freqs_original = compute_predicted_emotions_frequencies(predicted_emotions_original, version='Original')
    # shape {dataset: {speaker: {emotion: {pred_emotion: count}}}}
    freqs_baseline = compute_predicted_emotions_frequencies(predicted_emotions_baseline, version='Baseline')
    # shape {dataset: {speaker: {emotion: {pred_emotion: count}}}}
    freqs_sent = compute_predicted_emotions_frequencies(predicted_emotions_sent, version='Sent')
    # shape {dataset: {speaker: {prompt_emotion: {pred_emotion: count}}}}
    freqs_prompt = compute_predicted_emotions_frequencies(predicted_emotions_prompt, version='Prompt')

    torch.save(freqs_original, os.path.join(PREPROCESSING_DIR, "Evaluation", "freqs_original.pt"))
    torch.save(freqs_baseline, os.path.join(PREPROCESSING_DIR, "Evaluation", "freqs_baseline.pt"))
    torch.save(freqs_sent, os.path.join(PREPROCESSING_DIR, "Evaluation", "freqs_sent.pt"))
    torch.save(freqs_prompt, os.path.join(PREPROCESSING_DIR, "Evaluation", "freqs_prompt.pt"))
