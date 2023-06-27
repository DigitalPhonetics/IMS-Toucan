import os
from tqdm import tqdm
import csv
from numpy import trim_zeros
import string

import torch
import torchaudio
from torch.nn import CosineSimilarity
from datasets import load_dataset
import pandas as pd
import soundfile as sf

from Utility.storage_config import PREPROCESSING_DIR
from Utility.utils import get_emotion_from_path
from Utility.utils import float2pcm
from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface
from Preprocessing.AudioPreprocessor import AudioPreprocessor

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
        if version != "Original":
            speaker_embeddings[version] = {}
            for dataset in tqdm(os.listdir(os.path.join(audio_dir, version)), "Dataset"):
                speaker_embeddings[version][dataset] = {}
                for audio_file in tqdm(os.listdir(os.path.join(audio_dir, version, dataset)), "Audio File"):
                    emotion = audio_file.split("_")[0]
                    speaker = audio_file.split("_")[3].split(".wav")[0]
                    wave, sr = torchaudio.load(os.path.join(audio_dir, version, dataset, audio_file))
                    # mono
                    wave = torch.mean(wave, dim=0, keepdim=True)
                    # resampling
                    wave = torchaudio.functional.resample(wave, orig_freq=sr, new_freq=16000)
                    wave = wave.squeeze(0)
                    embedding = classifier.encode_batch(wave).squeeze(0).squeeze(0)
                    if speaker not in speaker_embeddings[version][dataset]:
                        speaker_embeddings[version][dataset][speaker] = {"anger":[], "joy":[], "neutral":[], "sadness":[], "surprise":[]}
                    speaker_embeddings[version][dataset][speaker][emotion].append(embedding)
    return speaker_embeddings

def extract_speaker_embeddings_original(audio_dir, classifier):
    speaker_embeddings = {}
    for version in os.listdir(audio_dir):
        if version == "Original":
            for speaker in tqdm(os.listdir(os.path.join(audio_dir, version)), "Speaker"):
                speaker_embeddings[speaker] = {}
                for emotion in os.listdir(os.path.join(audio_dir, version, speaker)):
                    speaker_embeddings[speaker][emotion] = []
                    for audio_file in os.listdir(os.path.join(audio_dir, version, speaker, emotion)):
                        wave, sr = torchaudio.load(os.path.join(audio_dir, version, speaker, emotion, audio_file))
                        # mono
                        wave = torch.mean(wave, dim=0, keepdim=True)
                        # resampling
                        wave = torchaudio.functional.resample(wave, orig_freq=sr, new_freq=16000)
                        wave = wave.squeeze(0)
                        embedding = classifier.encode_batch(wave).squeeze(0).squeeze(0)
                        speaker_embeddings[speaker][emotion].append(embedding)
    return speaker_embeddings

def vocode_original(mel2wav):
    esds_data_dir = "/mount/resources/speech/corpora/Emotional_Speech_Dataset_Singapore"
    emotion_to_sent_ids = {"anger":["000363", "000421", "000474"], 
                           "joy":["000713", "000771", "000824"], 
                           "neutral":["000013", "000071", "000124"], 
                           "sadness":["001063", "001121", "001174"], 
                           "surprise":["001413", "001471", "001524"]}
    for speaker in tqdm(os.listdir(esds_data_dir), "Speaker"):
        if speaker.startswith("00"):
            if int(speaker) > 10:
                for emotion in tqdm(os.listdir(os.path.join(esds_data_dir, speaker)), "Emotion"):
                    if not emotion.endswith(".txt") and not emotion.endswith(".DS_Store"):
                        for audio_file in os.listdir(os.path.join(esds_data_dir, speaker, emotion)):
                            if audio_file.endswith(".wav"):
                                emo = get_emotion_from_path(os.path.join(esds_data_dir, speaker, emotion, audio_file))
                                sent_id = audio_file.split("_")[1].split(".wav")[0]
                                if sent_id in emotion_to_sent_ids[emo]:
                                    wave, sr = sf.read(os.path.join(esds_data_dir, speaker, emotion, audio_file))
                                    ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024, cut_silence=True, device='cpu')
                                    norm_wave = ap.audio_to_wave_tensor(normalize=True, audio=wave)
                                    norm_wave = torch.tensor(trim_zeros(norm_wave.numpy()))
                                    spec = ap.audio_to_mel_spec_tensor(audio=norm_wave, normalize=False, explicit_sampling_rate=16000).cpu()

                                    wave = mel2wav(spec)
                                    silence = torch.zeros([10600])
                                    wav = silence.clone()
                                    wav = torch.cat((wav, wave, silence), 0)

                                    wav = [val for val in wav.detach().numpy() for _ in (0, 1)]  # doubling the sampling rate for better compatibility (24kHz is not as standard as 48kHz)
                                    os.makedirs(os.path.join(f"./audios/Evaluation/Original/{speaker}/{emo}"), exist_ok=True)
                                    sf.write(file=f"./audios/Evaluation/Original/{speaker}/{emo}/{sent_id}.wav", data=float2pcm(wav), samplerate=48000, subtype="PCM_16")
                        

def speaker_similarity(speaker_embedding1, speaker_embedding2):
    cosine_similarity = CosineSimilarity(dim=-1)
    return cosine_similarity(speaker_embedding1, speaker_embedding2)

def asr_transcribe(audio_dir, processor, model):
    transcriptions = {}
    for version in tqdm(os.listdir(audio_dir), "Version"):
        if version != "Original":
            transcriptions[version] = {}
            for dataset in tqdm(os.listdir(os.path.join(audio_dir, version)), "Dataset"):
                transcriptions[version][dataset] = {}
                for audio_file in tqdm(os.listdir(os.path.join(audio_dir, version, dataset)), "Audio File"):
                    emotion = audio_file.split("_")[0]
                    speaker = audio_file.split("_")[3].split(".wav")[0]
                    sentence_id = int(audio_file.split("_")[1])
                    wave, sr = torchaudio.load(os.path.join(audio_dir, version, dataset, audio_file))
                    # mono
                    wave = torch.mean(wave, dim=0, keepdim=True)
                    # resampling
                    wave = torchaudio.functional.resample(wave, orig_freq=sr, new_freq=16000)
                    wave = wave.squeeze(0)
                    input_values = processor(wave, sampling_rate=16000, return_tensors="pt", padding="longest").input_values.to(model.device)
                    with torch.no_grad():
                        logits = model(input_values).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    transcription = processor.batch_decode(predicted_ids)
                    if speaker not in transcriptions[version][dataset]:
                        transcriptions[version][dataset][speaker] = {"anger":{}, "joy":{}, "neutral":{}, "sadness":{}, "surprise":{}}
                    transcriptions[version][dataset][speaker][emotion][sentence_id] = transcription
    return transcriptions

def word_error_rate(target, predicted, wer):
    target = target.translate(str.maketrans('', '', string.punctuation)).upper()
    return float(wer(predicted, target))

def classify_speech_emotion(audio_dir, classifier):
    emotion_classified = {}
    for version in tqdm(os.listdir(audio_dir), "Version"):
        if version != "Original":
            emotion_classified[version] = {}
            for dataset in tqdm(os.listdir(os.path.join(audio_dir, version)), "Dataset"):
                emotion_classified[version][dataset] = {}
                for audio_file in tqdm(os.listdir(os.path.join(audio_dir, version, dataset)), "Audio File"):
                    emotion = audio_file.split("_")[0]
                    speaker = audio_file.split("_")[3].split(".wav")[0]
                    sentence_id = int(audio_file.split("_")[1])
                    out_prob, score, index, text_lab = classifier.classify_file(os.path.join(audio_dir, version, dataset, audio_file))
                    if speaker not in emotion_classified[version][dataset]:
                        emotion_classified[version][dataset][speaker] = {"anger":{}, "joy":{}, "neutral":{}, "sadness":{}, "surprise":{}}
                    emotion_classified[version][dataset][speaker][emotion][sentence_id] = text_lab[0]
    return emotion_classified