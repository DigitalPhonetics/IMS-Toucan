import os
from tqdm import tqdm
import csv
from numpy import trim_zeros
import string
import subprocess
import time
from statistics import median, mean

import torch
import torchaudio
from torch.nn import CosineSimilarity
from datasets import load_dataset
import pandas as pd
import soundfile as sf
from torchmetrics import WordErrorRate

from Utility.storage_config import PREPROCESSING_DIR
from Utility.utils import get_emotion_from_path
from Utility.utils import float2pcm
from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface
from Preprocessing.AudioPreprocessor import AudioPreprocessor

EMOTIONS = ["anger", "joy", "neutral", "sadness", "surprise"]

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
    if version == "Prompt":
        os.makedirs(f"audios/Evaluation/Prompt", exist_ok=True)
        os.makedirs(f"audios/Evaluation/Prompt/dailydialogues", exist_ok=True)
        os.makedirs(f"audios/Evaluation/Prompt/tales", exist_ok=True)
        model_id = "Sent_Finetuning_2_80k"

    tts = ToucanTTSInterface(device=exec_device, 
                             tts_model_path=model_id, 
                             vocoder_model_path=vocoder_model_path, 
                             faster_vocoder=not biggan, 
                             sent_emb_extractor=sent_emb_extractor)
    tts.set_language("en")
    for speaker_id in tqdm(range(25, 35), "Speakers"):
        os.makedirs(f"audios/Evaluation/{version}/dailydialogues/{speaker_id - 14}", exist_ok=True)
        os.makedirs(f"audios/Evaluation/{version}/tales/{speaker_id - 14}", exist_ok=True)
        tts.set_speaker_id(speaker_id)
        for dataset, emotion_to_sents in tqdm(test_sentences.items(), "Datasets"):
            for emotion, sentences in tqdm(emotion_to_sents.items(), "Emotions"):
                os.makedirs(f"audios/Evaluation/{version}/{dataset}/{speaker_id - 14}/{emotion}", exist_ok=True)
                for i, sent in enumerate(tqdm(sentences, "Sentences")):
                    if version == 'Prompt':
                        for prompt_emotion in list(emotion_to_sents.keys()):
                            os.makedirs(f"audios/Evaluation/{version}/{dataset}/{speaker_id - 14}/{emotion}/{prompt_emotion}", exist_ok=True)
                            prompt = emotion_to_sents[prompt_emotion][len(sentences) - 1 - i]
                            tts.set_sentence_embedding(prompt)
                            tts.read_to_file(text_list=[sent], 
                                        file_location=f"audios/Evaluation/{version}/{dataset}/{speaker_id - 14}/{emotion}/{prompt_emotion}/{i}.wav",
                                        increased_compatibility_mode=True,
                                        silent=silent)
                    else:
                        tts.read_to_file(text_list=[sent], 
                                        file_location=f"audios/Evaluation/{version}/{dataset}/{speaker_id - 14}/{emotion}/{i}.wav",
                                        increased_compatibility_mode=True,
                                        silent=silent)
                    
def extract_speaker_embeddings(audio_dir, classifier, version):
    speaker_embeddings = {}
    if version == "Original":
        for speaker in tqdm(os.listdir(os.path.join(audio_dir, version)), "Speaker"):
            speaker_embeddings[speaker] = {}
            for emotion in tqdm(os.listdir(os.path.join(audio_dir, version, speaker)), "Emotion"):
                speaker_embeddings[speaker][emotion] = {}
                for audio_file in tqdm(os.listdir(os.path.join(audio_dir, version, speaker, emotion)), "Audio File"):
                    file_id = int(audio_file.split('.wav')[0])
                    wave, sr = torchaudio.load(os.path.join(audio_dir, version, speaker, emotion, audio_file))
                    # mono
                    wave = torch.mean(wave, dim=0, keepdim=True)
                    # resampling
                    wave = torchaudio.functional.resample(wave, orig_freq=sr, new_freq=16000)
                    wave = wave.squeeze(0)
                    embedding = classifier.encode_batch(wave).squeeze(0).squeeze(0)
                    speaker_embeddings[speaker][emotion][file_id] = embedding
        return speaker_embeddings
    if version == "Baseline" or version == "Sent":
        for dataset in tqdm(os.listdir(os.path.join(audio_dir, version)), "Dataset"):
            speaker_embeddings[dataset] = {}
            for speaker in tqdm(os.listdir(os.path.join(audio_dir, version, dataset)), "Speaker"):
                speaker_embeddings[dataset][speaker] = {}
                for emotion in tqdm(os.listdir(os.path.join(audio_dir, version, dataset, speaker)), "Emotion"):
                    speaker_embeddings[dataset][speaker][emotion] = {}
                    for audio_file in tqdm(os.listdir(os.path.join(audio_dir, version, dataset, speaker, emotion)), "Audio File"):
                        file_id = int(audio_file.split('.wav')[0])
                        wave, sr = torchaudio.load(os.path.join(audio_dir, version, dataset, speaker, emotion, audio_file))
                        # mono
                        wave = torch.mean(wave, dim=0, keepdim=True)
                        # resampling
                        wave = torchaudio.functional.resample(wave, orig_freq=sr, new_freq=16000)
                        wave = wave.squeeze(0)
                        embedding = classifier.encode_batch(wave).squeeze(0).squeeze(0)
                        speaker_embeddings[dataset][speaker][emotion][file_id] = embedding
        return speaker_embeddings
    if version == "Prompt":
        for dataset in tqdm(os.listdir(os.path.join(audio_dir, version)), "Dataset"):
            speaker_embeddings[dataset] = {}
            for speaker in tqdm(os.listdir(os.path.join(audio_dir, version, dataset)), "Speaker"):
                speaker_embeddings[dataset][speaker] = {}
                for emotion in tqdm(os.listdir(os.path.join(audio_dir, version, dataset, speaker)), "Emotion"):
                    speaker_embeddings[dataset][speaker][emotion] = {}
                    for prompt_emotion in tqdm(os.listdir(os.path.join(audio_dir, version, dataset, speaker, emotion)), "Prompt Emotion"):
                        speaker_embeddings[dataset][speaker][emotion][prompt_emotion] = {}
                        for audio_file in tqdm(os.listdir(os.path.join(audio_dir, version, dataset, speaker, emotion, prompt_emotion)), "Audio File"):
                            file_id = int(audio_file.split('.wav')[0])
                            wave, sr = torchaudio.load(os.path.join(audio_dir, version, dataset, speaker, emotion, prompt_emotion, audio_file))
                            # mono
                            wave = torch.mean(wave, dim=0, keepdim=True)
                            # resampling
                            wave = torchaudio.functional.resample(wave, orig_freq=sr, new_freq=16000)
                            wave = wave.squeeze(0)
                            embedding = classifier.encode_batch(wave).squeeze(0).squeeze(0)
                            speaker_embeddings[dataset][speaker][emotion][prompt_emotion][file_id] = embedding
        return speaker_embeddings

def vocode_original(mel2wav, num_sentences, device):
    esds_data_dir = "/mount/resources/speech/corpora/Emotional_Speech_Dataset_Singapore"
    for speaker in tqdm(os.listdir(esds_data_dir), "Speaker"):
        if speaker.startswith("00"):
            if int(speaker) > 10:
                for emotion in tqdm(os.listdir(os.path.join(esds_data_dir, speaker)), "Emotion"):
                    if not emotion.endswith(".txt") and not emotion.endswith(".DS_Store"):
                        counter = 0
                        for audio_file in tqdm(os.listdir(os.path.join(esds_data_dir, speaker, emotion))):
                            if audio_file.endswith(".wav"):
                                counter += 1
                                if counter > num_sentences:
                                    break
                                emo = get_emotion_from_path(os.path.join(esds_data_dir, speaker, emotion, audio_file))
                                sent_id = audio_file.split("_")[1].split(".wav")[0]

                                wave, sr = sf.read(os.path.join(esds_data_dir, speaker, emotion, audio_file))
                                ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024, cut_silence=True, device='cpu')
                                norm_wave = ap.audio_to_wave_tensor(normalize=True, audio=wave)
                                norm_wave = torch.tensor(trim_zeros(norm_wave.numpy()))
                                spec = ap.audio_to_mel_spec_tensor(audio=norm_wave, normalize=False, explicit_sampling_rate=16000)

                                wave = mel2wav(spec.to(device)).cpu()
                                silence = torch.zeros([10600])
                                wav = silence.clone()
                                wav = torch.cat((wav, wave, silence), 0)

                                wav = [val for val in wav.detach().numpy() for _ in (0, 1)]  # doubling the sampling rate for better compatibility (24kHz is not as standard as 48kHz)
                                os.makedirs(os.path.join(f"./audios/Evaluation/Original/{int(speaker)}/{emo}"), exist_ok=True)
                                sf.write(file=f"./audios/Evaluation/Original/{int(speaker)}/{emo}/{sent_id}.wav", data=float2pcm(wav), samplerate=48000, subtype="PCM_16")
                        
def compute_speaker_similarity(speaker_embeddings_original, speaker_embeddings, version):
    speaker_similarities = {}
    if version == "Baseline" or version == "Sent":
        for dataset, speakers in tqdm(speaker_embeddings.items()):
            speaker_similarities[dataset] = {}
            for speaker, emotions in speakers.items():
                speaker_similarities[dataset][speaker] = {}
                for emotion, file_ids in emotions.items():
                    cos_sims_emotion = []
                    for file_id, embedding in file_ids.items():
                        cos_sims_file = []
                        for file_id_original, embedding_original in speaker_embeddings_original[speaker][emotion].items():
                            cos_sims_file.append(speaker_similarity(embedding_original, embedding))
                        cos_sims_emotion.append(median(cos_sims_file))
                    speaker_similarities[dataset][speaker][emotion] = median(cos_sims_emotion)
        return speaker_similarities
    if version == "Prompt":
        for dataset, speakers in tqdm(speaker_embeddings.items()):
            speaker_similarities[dataset] = {}
            for speaker, emotions in speakers.items():
                speaker_similarities[dataset][speaker] = {}
                for emotion, prompt_emotions in emotions.items():
                    speaker_similarities[dataset][speaker][emotion] = {}
                    for prompt_emotion, file_ids in prompt_emotions.items():
                        cos_sims_prompt_emotion = []
                        for file_id, embedding in file_ids.items():
                            cos_sims_file = []
                            for file_id_original, embedding_original in speaker_embeddings_original[speaker][emotion].items():
                                cos_sims_file.append(speaker_similarity(embedding_original, embedding))
                            cos_sims_prompt_emotion.append(median(cos_sims_file))
                        speaker_similarities[dataset][speaker][emotion][prompt_emotion] = median(cos_sims_prompt_emotion)

        speaker_similarities_prompt_emotions = {}
        for dataset, speakers in speaker_similarities.items():
            speaker_similarities_prompt_emotions[dataset] = {}
            for speaker, emotions in speakers.items():
                speaker_similarities_prompt_emotions[dataset][speaker] = {}
                for prompt_emotion in EMOTIONS:
                    cos_sims = []
                    for emotion in EMOTIONS:
                        cos_sims.append(speaker_similarities[dataset][speaker][emotion][prompt_emotion])
                    speaker_similarities_prompt_emotions[dataset][speaker][prompt_emotion] = median(cos_sims)
        return speaker_similarities_prompt_emotions

def speaker_similarity(speaker_embedding1, speaker_embedding2):
    cosine_similarity = CosineSimilarity(dim=-1)
    return cosine_similarity(speaker_embedding1, speaker_embedding2).numpy()

def asr_transcribe(audio_dir, processor, model, version):
    transcriptions = {}
    if version == "Original":
        for speaker in tqdm(os.listdir(os.path.join(audio_dir, version)), "Speaker"):
            transcriptions[speaker] = {}
            for emotion in tqdm(os.listdir(os.path.join(audio_dir, version, speaker)), "Emotion"):
                transcriptions[speaker][emotion] = {}
                for audio_file in tqdm(os.listdir(os.path.join(audio_dir, version, speaker, emotion)), "Audio File"):
                    file_id = int(audio_file.split('.wav')[0])
                    wave, sr = torchaudio.load(os.path.join(audio_dir, version, speaker, emotion, audio_file))
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
                    transcriptions[speaker][emotion][file_id] = transcription
        return transcriptions
    if version == "Baseline" or version == "Sent":
        for dataset in tqdm(os.listdir(os.path.join(audio_dir, version)), "Dataset"):
            transcriptions[dataset] = {}
            for speaker in tqdm(os.listdir(os.path.join(audio_dir, version, dataset)), "Speaker"):
                transcriptions[dataset][speaker] = {}
                for emotion in tqdm(os.listdir(os.path.join(audio_dir, version, dataset, speaker)), "Emotion"):
                    transcriptions[dataset][speaker][emotion] = {}
                    for audio_file in tqdm(os.listdir(os.path.join(audio_dir, version, dataset, speaker, emotion)), "Audio File"):
                        file_id = int(audio_file.split('.wav')[0])
                        wave, sr = torchaudio.load(os.path.join(audio_dir, version, dataset, speaker, emotion, audio_file))
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
                        transcriptions[dataset][speaker][emotion][file_id] = transcription
        return transcriptions
    if version == "Prompt":
        for dataset in tqdm(os.listdir(os.path.join(audio_dir, version)), "Dataset"):
            transcriptions[dataset] = {}
            for speaker in tqdm(os.listdir(os.path.join(audio_dir, version, dataset)), "Speaker"):
                transcriptions[dataset][speaker] = {}
                for emotion in tqdm(os.listdir(os.path.join(audio_dir, version, dataset, speaker)), "Emotion"):
                    transcriptions[dataset][speaker][emotion] = {}
                    for prompt_emotion in tqdm(os.listdir(os.path.join(audio_dir, version, dataset, speaker, emotion)), "Prompt Emotion"):
                        transcriptions[dataset][speaker][emotion][prompt_emotion] = {}
                        for audio_file in tqdm(os.listdir(os.path.join(audio_dir, version, dataset, speaker, emotion, prompt_emotion)), "Audio File"):
                            file_id = int(audio_file.split('.wav')[0])
                            wave, sr = torchaudio.load(os.path.join(audio_dir, version, dataset, speaker, emotion, prompt_emotion, audio_file))
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
                            transcriptions[dataset][speaker][emotion][prompt_emotion][file_id] = transcription
        return transcriptions
    
def compute_word_error_rate(transcriptions, test_sentences, version):
    wer_calc = WordErrorRate()
    word_error_rates = {}
    if version == "Original":
        for speaker, emotions in tqdm(transcriptions.items()):
            word_error_rates[speaker] = {}
            for emotion, sent_ids in emotions.items():
                wers = []
                for sent_id, transcript in sent_ids.items():
                    target = get_esds_target_transcript(speaker, sent_id)
                    wers.append(word_error_rate(target, transcript, wer_calc))
                word_error_rates[speaker][emotion] = median(wers)
        return word_error_rates
    if version == "Baseline" or version == "Sent":
        for dataset, speakers in tqdm(transcriptions.items()):
            word_error_rates[dataset] = {}
            for speaker, emotions in speakers.items():
                word_error_rates[dataset][speaker] = {}
                for emotion, sent_ids in emotions.items():
                    wers = []
                    for sent_id, transcript in sent_ids.items():
                        target = test_sentences[dataset][emotion][sent_id]
                        wers.append(word_error_rate(target, transcript, wer_calc))
                    word_error_rates[dataset][speaker][emotion] = median(wers)
        return word_error_rates
    if version == "Prompt":
        for dataset, speakers in tqdm(transcriptions.items()):
            word_error_rates[dataset] = {}
            for speaker, emotions in speakers.items():
                word_error_rates[dataset][speaker] = {}
                for emotion, prompt_emotions in emotions.items():
                    word_error_rates[dataset][speaker][emotion] = {}
                    for prompt_emotion, sent_ids in prompt_emotions.items():
                        wers = []
                        for sent_id, transcript in sent_ids.items():
                            target = test_sentences[dataset][emotion][sent_id]
                            wers.append(word_error_rate(target, transcript, wer_calc))
                        word_error_rates[dataset][speaker][emotion][prompt_emotion] = median(wers)
        word_error_rates_prompt_emotions = {}
        for dataset, speakers in word_error_rates.items():
            word_error_rates_prompt_emotions[dataset] = {}
            for speaker, emotions in speakers.items():
                word_error_rates_prompt_emotions[dataset][speaker] = {}
                for prompt_emotion in EMOTIONS:
                    wers = []
                    for emotion in EMOTIONS:
                        wers.append(word_error_rates[dataset][speaker][emotion][prompt_emotion])
                    word_error_rates_prompt_emotions[dataset][speaker][prompt_emotion] = median(wers)
        return word_error_rates_prompt_emotions
    
def get_esds_target_transcript(speaker, sent_id):
    sent_id = '0' * (6 - len(str(sent_id))) + str(sent_id) # insert zeros at the beginning
    root = "/mount/resources/speech/corpora/Emotional_Speech_Dataset_Singapore"
    speaker_dir = f"00{speaker}"
    with open(f"{root}/{speaker_dir}/fixed_unicode.txt", mode="r", encoding="utf8") as f:
        transcripts = f.read()
        for line in transcripts.replace("\n\n", "\n").replace(",", ", ").split("\n"):
            if line.strip() != "":
                filename, text, emo_dir = line.split("\t")
                if filename.split("_")[1] == sent_id:
                    return text

def word_error_rate(target, predicted, wer):
    target = target.translate(str.maketrans('', '', string.punctuation)).upper()
    return float(wer(predicted, target))

def classify_speech_emotion(audio_dir, classifier, version):
    emotions_classified = {}
    if version == "Original":
        for speaker in tqdm(os.listdir(os.path.join(audio_dir, version)), "Speaker"):
            emotions_classified[speaker] = {}
            for emotion in tqdm(os.listdir(os.path.join(audio_dir, version, speaker)), "Emotion"):
                emotions_classified[speaker][emotion] = {}
                for audio_file in tqdm(os.listdir(os.path.join(audio_dir, version, speaker, emotion)), "Audio File"):
                    file_id = int(audio_file.split('.wav')[0])
                    out_prob, score, index, text_lab = classifier.classify_file(os.path.join(audio_dir, version, speaker, emotion, audio_file))
                    emotions_classified[speaker][emotion][file_id] = text_lab[0]
                # wav2vec2 saves wav files, they have to be deleted such they are not used again for the next iteration, since they are named the same
                command = 'rm *.wav'
                process = subprocess.Popen(command, shell=True)
                process.wait()
                time.sleep(10) # ensure that files are deleted before next iteration
        return emotions_classified
    if version == "Baseline" or version == "Sent":
        for dataset in tqdm(os.listdir(os.path.join(audio_dir, version)), "Dataset"):
            emotions_classified[dataset] = {}
            for speaker in tqdm(os.listdir(os.path.join(audio_dir, version, dataset)), "Speaker"):
                emotions_classified[dataset][speaker] = {}
                for emotion in tqdm(os.listdir(os.path.join(audio_dir, version, dataset, speaker)), "Emotion"):
                    emotions_classified[dataset][speaker][emotion] = {}
                    for audio_file in tqdm(os.listdir(os.path.join(audio_dir, version, dataset, speaker, emotion)), "Audio File"):
                        file_id = int(audio_file.split('.wav')[0])
                        out_prob, score, index, text_lab = classifier.classify_file(os.path.join(audio_dir, version, dataset, speaker, emotion, audio_file))
                        emotions_classified[dataset][speaker][emotion][file_id] = text_lab[0]
                    # wav2vec2 saves wav files, they have to be deleted such they are not used again for the next iteration, since they are named the same
                    command = 'rm *.wav'
                    process = subprocess.Popen(command, shell=True)
                    process.wait()
                    time.sleep(10) # ensure that files are deleted before next iteration
        return emotions_classified
    if version == "Prompt":
        for dataset in tqdm(os.listdir(os.path.join(audio_dir, version)), "Dataset"):
            emotions_classified[dataset] = {}
            for speaker in tqdm(os.listdir(os.path.join(audio_dir, version, dataset)), "Speaker"):
                emotions_classified[dataset][speaker] = {}
                for emotion in tqdm(os.listdir(os.path.join(audio_dir, version, dataset, speaker)), "Emotion"):
                    emotions_classified[dataset][speaker][emotion] = {}
                    for prompt_emotion in tqdm(os.listdir(os.path.join(audio_dir, version, dataset, speaker, emotion)), "Prompt Emotion"):
                        emotions_classified[dataset][speaker][emotion][prompt_emotion] = {}
                        for audio_file in tqdm(os.listdir(os.path.join(audio_dir, version, dataset, speaker, emotion, prompt_emotion)), "Audio File"):
                            file_id = int(audio_file.split('.wav')[0])
                            out_prob, score, index, text_lab = classifier.classify_file(os.path.join(audio_dir, version, dataset, speaker, emotion, prompt_emotion, audio_file))
                            emotions_classified[dataset][speaker][emotion][prompt_emotion][file_id] = text_lab[0]
                        # wav2vec2 saves wav files, they have to be deleted such they are not used again for the next iteration, since they are named the same
                        command = 'rm *.wav'
                        process = subprocess.Popen(command, shell=True)
                        process.wait()
                        time.sleep(10) # ensure that files are deleted before next iteration
        return emotions_classified
    
def compute_predicted_emotions_frequencies(predicted_emotions, version):
    predicted_frequencies = {}
    if version == "Original":
        for speaker, emotions in tqdm(predicted_emotions.items()):
            predicted_frequencies[speaker] = {}
            for emotion, sent_ids in emotions.items():
                predicted_frequencies[speaker][emotion] = {}
                for sent_id, predicted_emotion in sent_ids.items():
                    if predicted_emotion not in predicted_frequencies[speaker][emotion]:
                        predicted_frequencies[speaker][emotion][predicted_emotion] = 0
                    predicted_frequencies[speaker][emotion][predicted_emotion] += 1
        return predicted_frequencies
    if version == "Baseline" or version == "Sent":
        for dataset, speakers in tqdm(predicted_emotions.items()):
            predicted_frequencies[dataset] = {}
            for speaker, emotions in speakers.items():
                predicted_frequencies[dataset][speaker] = {}
                for emotion, sent_ids in emotions.items():
                    predicted_frequencies[dataset][speaker][emotion] = {}
                    for sent_id, predicted_emotion in sent_ids.items():
                        if predicted_emotion not in predicted_frequencies[dataset][speaker][emotion]:
                            predicted_frequencies[dataset][speaker][emotion][predicted_emotion] = 0
                        predicted_frequencies[dataset][speaker][emotion][predicted_emotion] += 1
        return predicted_frequencies
    if version == "Prompt":
        for dataset, speakers in tqdm(predicted_emotions.items()):
            predicted_frequencies[dataset] = {}
            for speaker, emotions in speakers.items():
                predicted_frequencies[dataset][speaker] = {}
                for emotion, prompt_emotions in emotions.items():
                    predicted_frequencies[dataset][speaker][emotion] = {}
                    for prompt_emotion, sent_ids in prompt_emotions.items():
                        predicted_frequencies[dataset][speaker][emotion][prompt_emotion] = {}
                        for sent_id, predicted_emotion in sent_ids.items():
                            if predicted_emotion not in predicted_frequencies[dataset][speaker][emotion][prompt_emotion]:
                                predicted_frequencies[dataset][speaker][emotion][prompt_emotion][predicted_emotion] = 0
                            predicted_frequencies[dataset][speaker][emotion][prompt_emotion][predicted_emotion] += 1
        predicted_frequencies_prompt_emotions = {}
        for dataset, speakers in predicted_frequencies.items():
            predicted_frequencies_prompt_emotions[dataset] = {}
            for speaker, emotions in speakers.items():
                predicted_frequencies_prompt_emotions[dataset][speaker] = {}
                for prompt_emotion in EMOTIONS:
                    predicted_frequencies_prompt_emotions[dataset][speaker][prompt_emotion] = {}
                    for emotion in EMOTIONS:
                        pred_freqs = predicted_frequencies[dataset][speaker][emotion][prompt_emotion]
                        for pred_emo, freq in pred_freqs.items():
                            if pred_emo not in predicted_frequencies_prompt_emotions[dataset][speaker][prompt_emotion]:
                                predicted_frequencies_prompt_emotions[dataset][speaker][prompt_emotion][pred_emo] = freq
                            else:
                                predicted_frequencies_prompt_emotions[dataset][speaker][prompt_emotion][pred_emo] += freq
        return predicted_frequencies_prompt_emotions
    