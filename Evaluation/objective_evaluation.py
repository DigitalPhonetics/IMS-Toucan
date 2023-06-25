import os
from tqdm import tqdm
import csv

import torch
import torchaudio
from torch.nn import CosineSimilarity
from datasets import load_dataset
import pandas as pd

from Utility.storage_config import PREPROCESSING_DIR
from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface

def extract_dailydialogue_sentences():
    dataset = load_dataset("daily_dialog", split="train", cache_dir=os.path.join(PREPROCESSING_DIR, 'DailyDialogues'))
    id_to_emotion = {0: "neutral", 1: "anger", 2: "disgust", 3: "fear", 4: "joy", 5: "sadness", 6: "surprise"}
    emotion_to_sents = emotion_to_sents = {"anger":[], "disgust":[], "fear":[], "joy":[], "neutral":[], "sadness":[], "surprise":[]}

    for dialog, emotions in tqdm(zip(dataset["dialog"], dataset["emotion"])):
        for sent, emotion in zip(dialog, emotions):
            emotion_to_sents[id_to_emotion[emotion]].append(sent.strip())

    return emotion_to_sents

def extract_tales_sentences(data_dir):
    id_to_emotion = {"N": "neutral", "A": "anger", "D": "disgust", "F": "fear", "H": "joy", "Sa": "sadness", "Su+": "surprise", "Su-": "surprise"}
    emotion_to_sents = emotion_to_sents = {"anger":[], "disgust":[], "fear":[], "joy":[], "neutral":[], "sadness":[], "surprise":[]}
    
    for author in tqdm(os.listdir(data_dir)):
        if not author.endswith(".pt"):
            for file in os.listdir(os.path.join(data_dir, author, "emmood")):
                df = pd.read_csv(os.path.join(data_dir, author, "emmood", file), sep="\t", header=None, quoting=csv.QUOTE_NONE)
                for index, (sent_id, emo, mood, sent) in df.iterrows():
                    emotions = emo.split(":")
                    if emotions[0] == emotions[1]:
                        emotion_to_sents[id_to_emotion[emotions[0]]].append(sent)
    return emotion_to_sents

def get_sorted_test_sentences(emotion_to_sents, classifier):
    emotion_to_sents_sorted = {}
    for emotion, sentences in emotion_to_sents.items():
        if emotion == "disgust" or emotion == "fear":
            continue
        sent_score = {}
        for sent in tqdm(sentences):
            result = classifier(sent)
            emo = result[0][0]['label']
            score = result[0][0]['score']
            if emo == emotion:
                sent_score[sent] = score
        sent_score = dict(sorted(sent_score.items(), key=lambda item: item[1], reverse=True))
        emotion_to_sents_sorted[emotion] = list(sent_score.keys())
    return emotion_to_sents_sorted

def synthesize_test_sentences(version="Baseline",
                              exec_device="cpu",
                              vocoder_model_path=None, 
                              biggan=False, 
                              sent_emb_extractor=None,
                              test_sentences=None,
                              silent=False):
    os.makedirs("audios", exist_ok=True)
    os.makedirs(f"audios/Evaluation", exist_ok=True)

    if version == "Baseline":
        os.makedirs(f"audios/Evaluation/Baseline", exist_ok=True)
        os.makedirs(f"audios/Evaluation/Baseline/dailydialogues", exist_ok=True)
        os.makedirs(f"audios/Evaluation/Baseline/tales", exist_ok=True)
        model_id = "Baseline_Finetuning_2_80k"
    if version == "Sent":
        os.makedirs(f"audios/Evaluation/Sent", exist_ok=True)
        os.makedirs(f"audios/Evaluation/Sent/dailydialogues", exist_ok=True)
        os.makedirs(f"audios/Evaluation/Sent/tales", exist_ok=True)
        model_id = "Sent_Finetuning_2_80k"

    tts = ToucanTTSInterface(device=exec_device, 
                             tts_model_path=model_id, 
                             vocoder_model_path=vocoder_model_path, 
                             faster_vocoder=not biggan, 
                             sent_emb_extractor=sent_emb_extractor)
    tts.set_language("en")
    for speaker_id in tqdm(range(25, 35), "Speakers"):
        tts.set_speaker_id(speaker_id)
        for dataset, emotion_to_sents in tqdm(test_sentences.items(), "Datasets"):
            for emotion, sentences in tqdm(emotion_to_sents.items(), "Emotions"):
                for i, sent in enumerate(tqdm(sentences, "Sentences")):
                    tts.read_to_file(text_list=[sent], 
                                    file_location=f"audios/Evaluation/{version}/{dataset}/{emotion}_{i}_ESDS_00{speaker_id - 14}.wav",
                                    increased_compatibility_mode=True,
                                    silent=silent)
                    
def extract_speaker_embeddings(audio_dir, classifier):
    speaker_embeddings = {}
    for version in tqdm(os.listdir(audio_dir), "Version"):
        speaker_embeddings[version] = {}
        for dataset in tqdm(os.listdir(os.path.join(audio_dir, version)), "Dataset"):
            speaker_embeddings[version][dataset] = {}
            for audio_file in tqdm(os.listdir(os.path.join(audio_dir, version, dataset)), "Audio File"):
                emotion = audio_file.split("_")[0]
                speaker = audio_file.split("_")[3]
                wave, sr = torchaudio.load(os.path.join(audio_dir, version, dataset, audio_file))
                # mono
                wave = torch.mean(wave, dim=0, keepdim=True)
                # resampling
                wave = torchaudio.functional.resample(wave, orig_freq=sr, new_freq=16000)
                wave = wave.squeeze(0)
                embedding = classifier.encode_batch(wave).squeeze(0).squeeze(0)
                if speaker not in speaker_embeddings[version][dataset]:
                    speaker_embeddings[version][dataset][speaker] = {"anger":[], "joy":[], "neutral":[], "sadness":[], "surprise":[]}
                else:
                    speaker_embeddings[version][dataset][speaker][emotion].append(embedding)
    return speaker_embeddings

def speaker_similarity(speaker_embedding1, speaker_embedding2):
    cosine_similarity = CosineSimilarity(dim=-1)
    return cosine_similarity(speaker_embedding1, speaker_embedding2)