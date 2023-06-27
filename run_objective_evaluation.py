import os
import argparse
from statistics import median, mean

import torch
from transformers import pipeline
from speechbrain.pretrained import EncoderClassifier
from speechbrain.pretrained.interfaces import foreign_class
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from torchmetrics import WordErrorRate

from Utility.storage_config import PREPROCESSING_DIR, MODELS_DIR
from Evaluation.objective_evaluation import *
from Preprocessing.sentence_embeddings.EmotionRoBERTaSentenceEmbeddingExtractor import EmotionRoBERTaSentenceEmbeddingExtractor
from InferenceInterfaces.InferenceArchitectures.InferenceAvocodo import HiFiGANGenerator

NUM_TEST_SENTENCES = 50 # 50 sentences per emotion category

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
    
    # get vocoded original sentences
    if not os.path.exists(os.path.join("./audios/Evaluation/Original")):
        print("Vocoding Original...")
        os.makedirs(os.path.join("./audios/Evaluation/Original"), exist_ok=True)
        vocoder_model_path = os.path.join(MODELS_DIR, "Avocodo", "best.pt")
        mel2wav = HiFiGANGenerator(path_to_weights=vocoder_model_path).to(torch.device('cpu'))
        mel2wav.remove_weight_norm()
        mel2wav.eval()
        vocode_original(mel2wav)
    
    # extract speaker embeddings
    if not os.path.exists(os.path.join(PREPROCESSING_DIR, "Evaluation", "speaker_embeddings.pt")):
        print("Extracting speaker embeddings...")
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", 
                                                    savedir="./Models/Embedding/spkrec-xvect-voxceleb", 
                                                    run_opts={"device": device})
        # shape {version: {dataset: {speaker: {emotion: [embeddings]}}}}
        speaker_embeddings = extract_speaker_embeddings("./audios/Evaluation", classifier)
        torch.save(speaker_embeddings, os.path.join(PREPROCESSING_DIR, "Evaluation", "speaker_embeddings.pt"))
        # shape {speaker: {emotion: [embeddings]}}
        speaker_embeddings_original = extract_speaker_embeddings_original("./audios/Evaluation", classifier)
        torch.save(speaker_embeddings_original, os.path.join(PREPROCESSING_DIR, "Evaluation", "speaker_embeddings_original.pt"))
    else:
        speaker_embeddings = torch.load(os.path.join(PREPROCESSING_DIR, "Evaluation", "speaker_embeddings.pt"), map_location='cpu')
        speaker_embeddings_original = torch.load(os.path.join(PREPROCESSING_DIR, "Evaluation", "speaker_embeddings_original.pt"), map_location='cpu')

    # calculate speaker similarity
    print("Calculating speaker similarity...")
    # shape {version: {dataset: {speaker: {emotion: speaker similarity}}}}
    speaker_similarities = {}
    for version, datasets in tqdm(speaker_embeddings.items()):
        speaker_similarities[version] = {}
        for dataset, speakers in datasets.items():
            speaker_similarities[version][dataset] = {}
            for speaker, emotions in speakers.items():
                speaker_similarities[version][dataset][speaker] = {}
                for emotion, embeddings in emotions.items():
                    cosine_similarities = []
                    for embedding in embeddings:
                        cosine_similarity = []
                        for embedding_original in speaker_embeddings_original[speaker][emotion]:
                            cosine_similarity.append(speaker_similarity(embedding_original, embedding))
                        cosine_similarity = median(cosine_similarity)
                        cosine_similarities.append(cosine_similarity)
                    speaker_similarities[version][dataset][speaker][emotion] = median(cosine_similarities)

    # calculate word error rate
    print("Calculating word error rate...")
    if not os.path.exists(os.path.join(PREPROCESSING_DIR, "Evaluation", "transcriptions.pt")):
        print("Transcribing...")
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", cache_dir=os.path.join(MODELS_DIR, "ASR"))
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", cache_dir=os.path.join(MODELS_DIR, "ASR")).to(device)
        # shape {version: {dataset: {speaker: {emotion: {sentence_id: transcription}}}}}
        transcriptions = asr_transcribe("./audios/Evaluation", processor, model)
        torch.save(transcriptions, os.path.join(PREPROCESSING_DIR, "Evaluation", "transcriptions.pt"))
    else:
        transcriptions = torch.load(os.path.join(PREPROCESSING_DIR, "Evaluation", "transcriptions.pt"), map_location='cpu')
    # shape {version: {dataset: {speaker: {emotion: word error rate}}}}
    wer_calc = WordErrorRate()
    wers = {}
    for version, datasets in tqdm(transcriptions.items()):
        wers[version] = {}
        for dataset, speakers in datasets.items():
            wers[version][dataset] = {}
            for speaker, emotions in speakers.items():
                wers[version][dataset][speaker] = {}
                for emotion, sentence_ids in emotions.items():
                    wer = []
                    for sentence_id, transcript in sentence_ids.items():
                        wer.append(word_error_rate(test_sentences[dataset][emotion][sentence_id], transcript, wer_calc))
                        wers[version][dataset][speaker][emotion] = median(wer)

    # speech emotion recognition
    if not os.path.exists(os.path.join(PREPROCESSING_DIR, "Evaluation", "predicted_emotions.pt")):
        print("Speech emotion recognition...")
        classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", 
                                pymodule_file="custom_interface.py", 
                                classname="CustomEncoderWav2vec2Classifier", 
                                savedir="./Models/Emotion_Recognition",
                                run_opts={"device":device})
        # shape {version: {dataset: {speaker: {emotion: {sentence_id: predicted emotion}}}}}
        predicted_emotions = classify_speech_emotion("./audios/Evaluation", classifier)
        torch.save(predicted_emotions, os.path.join(PREPROCESSING_DIR, "Evaluation", "predicted_emotions.pt"))
    else:
        predicted_emotions = torch.load(os.path.join(PREPROCESSING_DIR, "Evaluation", "predicted_emotions.pt"), map_location='cpu')

    print(predicted_emotions)
