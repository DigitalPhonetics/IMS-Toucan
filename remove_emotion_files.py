from tqdm import tqdm

from Utility.storage_config import PREPROCESSING_DIR
from Utility.corpus_preparation import prepare_fastspeech_corpus
from Utility.path_to_transcript_dicts import *

def get_emotion_from_path(path):
    if "EmoV_DB" in path:
        emotion = os.path.splitext(os.path.basename(path))[0].split("-16bit")[0].split("_")[0].lower()
        if emotion == "amused":
            emotion = "joy"
    if "CREMA_D" in path:
        emotion = os.path.splitext(os.path.basename(path))[0].split('_')[2]
        if emotion == "ANG":
            emotion = "anger"
        if emotion == "DIS":
            emotion = "disgust"
        if emotion == "FEA":
            emotion = "fear"
        if emotion == "HAP":
            emotion = "joy"
        if emotion == "NEU":
            emotion = "neutral"
        if emotion == "SAD":
            emotion = "sadness"
    if "Emotional_Speech_Dataset_Singapore" in path:
        emotion = os.path.basename(os.path.dirname(path)).lower()
        if emotion == "angry":
            emotion = "anger"
        if emotion == "happy":
            emotion = "joy"
        if emotion == "sad":
            emotion = "sadness"
    if "RAVDESS" in path:
        emotion = os.path.splitext(os.path.basename(path))[0].split('-')[2]
        if emotion == "01":
            emotion = "neutral"
        if emotion == "02":
            emotion = "calm"
        if emotion == "03":
            emotion = "joy"
        if emotion == "04":
            emotion = "sadness"
        if emotion == "05":
            emotion = "anger"
        if emotion == "06":
            emotion = "fear"
        if emotion == "07":
            emotion = "disgust"
        if emotion == "08":
            emotion = "surprise"
    return emotion

if __name__ == '__main__':
    train_set = prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_RAVDESS(),
                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "ravdess"),
                                          lang="en",
                                          save_imgs=False)
    
    remove_ids = []
    for index in tqdm(range(len(train_set))):
        path = train_set[index][10]
        emotion = get_emotion_from_path(path)
        if emotion == "sleepiness" or emotion == "calm":
            remove_ids.append(index)

    for remove_id in sorted(remove_ids, reverse=True):
        print(train_set[remove_id][10])

    #train_set.remove_samples(remove_ids)
