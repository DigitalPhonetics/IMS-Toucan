import os
import time

import torch

from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface
from Utility.storage_config import PREPROCESSING_DIR

def test_sentence(version, 
                  model_id="Meta", 
                  exec_device="cpu", 
                  speaker_reference=None, 
                  vocoder_model_path=None, 
                  biggan=False, 
                  sent_emb_extractor=None,
                  word_emb_extractor=None,
                  prompt:str=None,
                  xvect_model=None,
                  speaker_id=None):
    os.makedirs("audios", exist_ok=True)
    os.makedirs(f"audios/{version}", exist_ok=True)
    tts = ToucanTTSInterface(device=exec_device, 
                             tts_model_path=model_id, 
                             vocoder_model_path=vocoder_model_path, 
                             faster_vocoder=not biggan, 
                             sent_emb_extractor=sent_emb_extractor,
                             word_emb_extractor=word_emb_extractor, 
                             xvect_model=xvect_model)
    tts.set_language("en")
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if speaker_id is not None:
        tts.set_speaker_id(speaker_id)
    if prompt is not None:
        tts.set_sentence_embedding(prompt)

    #sentence = "The football teams give a tea party."
    sentence1 = "You can write an email."
    sentence2 = "They will arrive tomorrow."
    tts.read_to_file(text_list=[sentence1], 
                     file_location=f"audios/{version}/paper1.wav", 
                     increased_compatibility_mode=True, 
                     view_contours=True,
                     plot_name="paper1")
    start_time = time.time()
    tts.read_to_file(text_list=[sentence2], 
                     file_location=f"audios/{version}/paper2.wav", 
                     increased_compatibility_mode=True, 
                     view_contours=False,
                     plot_name="paper2")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")

def test_tales_emotion(version, model_id="Meta", 
                      exec_device="cpu", 
                      speaker_reference=None, 
                      vocoder_model_path=None, 
                      biggan=False, 
                      sent_emb_extractor=None, 
                      word_emb_extractor=None, 
                      prompt:str=None, 
                      xvect_model=None, 
                      speaker_id=None):
    os.makedirs("audios", exist_ok=True)
    os.makedirs(f"audios/{version}", exist_ok=True)
    os.makedirs(f"audios/{version}/Tales", exist_ok=True)
    tts = ToucanTTSInterface(device=exec_device, 
                             tts_model_path=model_id, 
                             vocoder_model_path=vocoder_model_path, 
                             faster_vocoder=not biggan, 
                             sent_emb_extractor=sent_emb_extractor, 
                             word_emb_extractor=word_emb_extractor, 
                             xvect_model=xvect_model)
    tts.set_language("en")
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if speaker_id is not None:
        tts.set_speaker_id(speaker_id)
    if prompt is not None:
        tts.set_sentence_embedding(prompt)

    emotion_to_sents = torch.load(os.path.join(PREPROCESSING_DIR, "Tales", f"emotion_sentences_top20.pt"), map_location='cpu')
    for emotion, sents in emotion_to_sents.items():
        for i, sent in enumerate(sents):
            tts.read_to_file(text_list=[sent], file_location=f"audios/{version}/Tales/{emotion}_{i}.wav", increased_compatibility_mode=True)

def test_yelp_emotion(version, model_id="Meta", 
                      exec_device="cpu", 
                      speaker_reference=None, 
                      vocoder_model_path=None, 
                      biggan=False, 
                      sent_emb_extractor=None, 
                      word_emb_extractor=None, 
                      prompt:str=None, 
                      xvect_model=None, 
                      speaker_id=None):
    os.makedirs("audios", exist_ok=True)
    os.makedirs(f"audios/{version}", exist_ok=True)
    os.makedirs(f"audios/{version}/Yelp", exist_ok=True)
    os.makedirs(f"audios/{version}/Yelp_Prompt", exist_ok=True)
    tts = ToucanTTSInterface(device=exec_device, 
                             tts_model_path=model_id, 
                             vocoder_model_path=vocoder_model_path, 
                             faster_vocoder=not biggan, 
                             sent_emb_extractor=sent_emb_extractor, 
                             word_emb_extractor=word_emb_extractor, 
                             xvect_model=xvect_model)
    tts.set_language("en")
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if speaker_id is not None:
        tts.set_speaker_id(speaker_id)
    if prompt is not None:
        tts.set_sentence_embedding(prompt)

    emotion_to_sents = torch.load(os.path.join(PREPROCESSING_DIR, "Yelp", f"emotion_sentences_top20.pt"), map_location='cpu')
    for emotion, sents in emotion_to_sents.items():
        for i, sent in enumerate(sents):
            tts.read_to_file(text_list=[sent], file_location=f"audios/{version}/Yelp/{emotion}_{i}.wav", increased_compatibility_mode=True)

def test_gne_emotion(version, model_id="Meta", 
                      exec_device="cpu", 
                      speaker_reference=None, 
                      vocoder_model_path=None, 
                      biggan=False, 
                      sent_emb_extractor=None, 
                      word_emb_extractor=None, 
                      prompt:str=None, 
                      xvect_model=None, 
                      speaker_id=None):
    os.makedirs("audios", exist_ok=True)
    os.makedirs(f"audios/{version}", exist_ok=True)
    os.makedirs(f"audios/{version}/Headlines", exist_ok=True)
    tts = ToucanTTSInterface(device=exec_device, 
                             tts_model_path=model_id, 
                             vocoder_model_path=vocoder_model_path, 
                             faster_vocoder=not biggan, 
                             sent_emb_extractor=sent_emb_extractor, 
                             word_emb_extractor=word_emb_extractor, 
                             xvect_model=xvect_model)
    tts.set_language("en")
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if speaker_id is not None:
        tts.set_speaker_id(speaker_id)
    if prompt is not None:
        tts.set_sentence_embedding(prompt)

    emotion_to_sents = torch.load(os.path.join(PREPROCESSING_DIR, "Headlines", f"emotion_sentences_top20.pt"), map_location='cpu')
    for emotion, sents in emotion_to_sents.items():
        for i, sent in enumerate(sents):
            tts.read_to_file(text_list=[sent], file_location=f"audios/{version}/Headlines/{emotion}_{i}.wav", increased_compatibility_mode=True)

def test_controllable(version, model_id="Meta", 
                      exec_device="cpu", 
                      speaker_reference=None, 
                      vocoder_model_path=None, 
                      biggan=False, 
                      sent_emb_extractor=None, 
                      word_emb_extractor=None, 
                      prompt:str=None, 
                      xvect_model=None, 
                      speaker_id=None):
    os.makedirs("audios", exist_ok=True)
    os.makedirs(f"audios/{version}", exist_ok=True)
    tts = ToucanTTSInterface(device=exec_device, 
                             tts_model_path=model_id, 
                             vocoder_model_path=vocoder_model_path, 
                             faster_vocoder=not biggan, 
                             sent_emb_extractor=sent_emb_extractor, 
                             word_emb_extractor=word_emb_extractor, 
                             xvect_model=xvect_model)
    tts.set_language("en")
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if speaker_id is not None:
        tts.set_speaker_id(speaker_id)
    if prompt is not None:
        tts.set_sentence_embedding(prompt)

    for i, sentence in enumerate(['I am so happy to see you!',
                                  'Today is a beautiful day and the sun is shining.',
                                  'He seemed to be quite lucky as he was smiling at me.',
                                  'She laughed and said: This is so funny.',
                                  'No, this is horrible!',
                                  'I am so sad, why is this so depressing?',
                                  'Be careful, cried the woman.',
                                  'This makes me feel bad.',
                                  'Oh happy day!',
                                  'Well, this sucks.',
                                  'That smell is disgusting.',
                                  'I am so angry!',
                                  'What a surprise!',
                                  'I am so scared, I fear the worst.',
                                  'This is a neutral test sentence with medium length, which should have relatively neutral prosody, and can be used to test the controllability through textual prompts.']):
        tts.read_to_file(text_list=[sentence], file_location=f"audios/{version}/Controllable_{i}.wav", increased_compatibility_mode=True)

def test_study(version, model_id="Meta", 
                      exec_device="cpu", 
                      speaker_reference=None, 
                      vocoder_model_path=None, 
                      biggan=False, 
                      sent_emb_extractor=None, 
                      word_emb_extractor=None, 
                      prompt:str=None, 
                      xvect_model=None, 
                      speaker_id=None):
    os.makedirs("audios", exist_ok=True)
    os.makedirs(f"audios/{version}", exist_ok=True)
    os.makedirs(f"audios/{version}/Study", exist_ok=True)
    tts = ToucanTTSInterface(device=exec_device, 
                             tts_model_path=model_id, 
                             vocoder_model_path=vocoder_model_path, 
                             faster_vocoder=not biggan, 
                             sent_emb_extractor=sent_emb_extractor, 
                             word_emb_extractor=word_emb_extractor, 
                             xvect_model=xvect_model)
    tts.set_language("en")
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if speaker_id is not None:
        tts.set_speaker_id(speaker_id)
    if prompt is not None:
        tts.set_sentence_embedding(prompt)

    emotion_to_sents = {"anger":   ["You can't be serious, how dare you not tell me you were going to marry her?",
                                    "I'm so angry, I feel like killing someone!",
                                    "It’s infuriating, I have to be in Rome by five!",
                                    "The bear, in great fury, ran after the carriage."
                                    ],
                        "disgust": ["I can't help myself, it's just so disgusting in here!",
                                    "What a stink, this place stinks like rotten eggs.",
                                    "I hate to complain, but this soup is too salty.",
                                    "The rabbits could not bear him, they could smell him half a mile off."
                                    ],
                        "sadness": ["I know that , mom, but sometimes I'm just sad.",
                                    "My uncle passed away last night.",
                                    "Lily broke up with me last week, in fact, she dumped me.",
                                    "Here he remained the whole night, feeling very tired and sorrowful."
                                    ],
                        "joy":     ["I am very happy to know my work could be recognized by you and our company.",
                                    "I really enjoy the beach in the summer.",
                                    "I had a wonderful time.",
                                    "Then she saw that her deliverance was near, and her heart leapt with joy."
                                    ],
                        "surprise":["I’m shocked he actually won.",
                                    "Wow, why so much, I thought they were getting you an assistant.",
                                    "Really, I can't believe it, it's like a dream come true, I never expected that I would win The Nobel Prize!",
                                    "He was astonished when he saw them come alone, and asked what had happened to them."
                                    ],
                        "fear":    ["I feel very nervous about it.",
                                    "I'm scared that she might not come back.",
                                    "Well , I just saw a horror movie last night, it almost frightened me to death.",
                                    "Peter sat down to rest, he was out of breath and trembling with fright, and he had not the least idea which way to go."
                                    ],
                        "neutral": ["You must specify an address of the place where you will spend most of your time.",
                                    "Just a second, I'll see if I can find them for you.",
                                    "You can go to the Employment Development Office and pick it up.",
                                    "So the queen gave him the letter, and said that he might see for himself what was written in it."
                                    ]
                        }
    for emotion, sents in emotion_to_sents.items():
        for i, sent in enumerate(sents):
            tts.read_to_file(text_list=[sent], file_location=f"audios/{version}/Study/{emotion}_{i}.wav", increased_compatibility_mode=True)

def test_study2(version, model_id="Meta", 
                      exec_device="cpu", 
                      speaker_reference=None, 
                      vocoder_model_path=None, 
                      biggan=False, 
                      sent_emb_extractor=None, 
                      word_emb_extractor=None, 
                      prompt:str=None, 
                      xvect_model=None, 
                      speaker_id=None):
    os.makedirs("audios", exist_ok=True)
    os.makedirs(f"audios/{version}", exist_ok=True)
    os.makedirs(f"audios/{version}/Study", exist_ok=True)
    tts = ToucanTTSInterface(device=exec_device, 
                             tts_model_path=model_id, 
                             vocoder_model_path=vocoder_model_path, 
                             faster_vocoder=not biggan, 
                             sent_emb_extractor=sent_emb_extractor, 
                             word_emb_extractor=word_emb_extractor, 
                             xvect_model=xvect_model)
    tts.set_language("en")
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if speaker_id is not None:
        tts.set_speaker_id(speaker_id)
    if prompt is not None:
        tts.set_sentence_embedding(prompt)

    emotion_to_sents = {"anger":   ["You can't be serious, how dare you not tell me you were going to marry her?",
                                    "The king grew angry, and cried: That is not allowed, he must appear before me and tell his name!"
                                    ],
                        "disgust": ["What a stink, this place stinks like rotten eggs.",
                                    "The rabbits could not bear him, they could smell him half a mile off."
                                    ],
                        "sadness": ["Lily broke up with me last week, in fact, she dumped me.",
                                    "The sisters mourned as young hearts can mourn, and were especially grieved at the sight of their parents' sorrow."
                                    ],
                        "joy":     ["I really enjoy the beach in the summer.",
                                    "Then she saw that her deliverance was near, and her heart leapt with joy."
                                    ],
                        "surprise":["Really? I can't believe it! It's like a dream come true, I never expected that I would win The Nobel Prize!",
                                    "He was astonished when he saw them come alone, and asked what had happened to them."
                                    ],
                        "fear":    ["I'm scared that she might not come back.",
                                    "Peter sat down to rest, he was out of breath and trembling with fright, and he had not the least idea which way to go."
                                    ],
                        "neutral": ["You can go to the Employment Development Office and pick it up.",
                                    "So the queen gave him the letter, and said that he might see for himself what was written in it."
                                    ]
                        }
    for emotion, sents in emotion_to_sents.items():
        for i, sent in enumerate(sents):
            tts.read_to_file(text_list=[sent], file_location=f"audios/{version}/Study/{emotion}_{i}.flac", increased_compatibility_mode=True)

def test_study2_male(version, model_id="Meta", 
                      exec_device="cpu", 
                      speaker_reference=None, 
                      vocoder_model_path=None, 
                      biggan=False, 
                      sent_emb_extractor=None, 
                      word_emb_extractor=None, 
                      prompt:str=None, 
                      xvect_model=None, 
                      speaker_id=None):
    os.makedirs("audios", exist_ok=True)
    os.makedirs(f"audios/{version}", exist_ok=True)
    os.makedirs(f"audios/{version}/Study", exist_ok=True)
    tts = ToucanTTSInterface(device=exec_device, 
                             tts_model_path=model_id, 
                             vocoder_model_path=vocoder_model_path, 
                             faster_vocoder=not biggan, 
                             sent_emb_extractor=sent_emb_extractor, 
                             word_emb_extractor=word_emb_extractor, 
                             xvect_model=xvect_model)
    tts.set_language("en")
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if speaker_id is not None:
        tts.set_speaker_id(speaker_id)

        emotion = "anger"
        sent = "The king grew angry, and cried: That is not allowed, he must appear before me and tell his name!"
        tts.read_to_file(text_list=[sent], file_location=f"audios/{version}/Study/sent_{emotion}_{1}.flac", increased_compatibility_mode=True)

        emotion = "joy"
        sent = "Then she saw that her deliverance was near, and her heart leapt with joy."
        tts.read_to_file(text_list=[sent], file_location=f"audios/{version}/Study/sent_{emotion}_{1}.flac", increased_compatibility_mode=True)

        emotion = "neutral"
        sent = "So the queen gave him the letter, and said that he might see for himself what was written in it."
        tts.read_to_file(text_list=[sent], file_location=f"audios/{version}/Study/sent_{emotion}_{1}.flac", increased_compatibility_mode=True)

        emotion = "sadness"
        sent = "The sisters mourned as young hearts can mourn, and were especially grieved at the sight of their parents' sorrow."
        tts.read_to_file(text_list=[sent], file_location=f"audios/{version}/Study/sent_{emotion}_{1}.flac", increased_compatibility_mode=True)

        emotion = "surprise"
        sent = "Really? I can't believe it! It's like a dream come true, I never expected that I would win The Nobel Prize!"
        tts.read_to_file(text_list=[sent], file_location=f"audios/{version}/Study/sent_{emotion}_{0}.flac", increased_compatibility_mode=True)

def test_study2_male_prompt(version, model_id="Meta", 
                      exec_device="cpu", 
                      speaker_reference=None, 
                      vocoder_model_path=None, 
                      biggan=False, 
                      sent_emb_extractor=None, 
                      word_emb_extractor=None, 
                      prompt:str=None, 
                      xvect_model=None, 
                      speaker_id=None):
    os.makedirs("audios", exist_ok=True)
    os.makedirs(f"audios/{version}", exist_ok=True)
    os.makedirs(f"audios/{version}/Study", exist_ok=True)
    tts = ToucanTTSInterface(device=exec_device, 
                             tts_model_path=model_id, 
                             vocoder_model_path=vocoder_model_path, 
                             faster_vocoder=not biggan, 
                             sent_emb_extractor=sent_emb_extractor, 
                             word_emb_extractor=word_emb_extractor, 
                             xvect_model=xvect_model)
    tts.set_language("en")
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if speaker_id is not None:
        tts.set_speaker_id(speaker_id)

        emotion = "anger"
        sent = "The king grew angry, and cried: That is not allowed, he must appear before me and tell his name!"
        prompt = "Really? I can't believe it! It's like a dream come true, I never expected that I would win The Nobel Prize!"
        tts.set_sentence_embedding(prompt)
        tts.read_to_file(text_list=[sent], file_location=f"audios/{version}/Study/sent_prompt_{emotion}_{1}.flac", increased_compatibility_mode=True)

        emotion = "joy"
        sent = "Then she saw that her deliverance was near, and her heart leapt with joy."
        prompt = "The sisters mourned as young hearts can mourn, and were especially grieved at the sight of their parents' sorrow."
        tts.set_sentence_embedding(prompt)
        tts.read_to_file(text_list=[sent], file_location=f"audios/{version}/Study/sent_prompt_{emotion}_{1}.flac", increased_compatibility_mode=True)

        emotion = "neutral"
        sent = "So the queen gave him the letter, and said that he might see for himself what was written in it."
        prompt = "The king grew angry, and cried: That is not allowed, he must appear before me and tell his name!"
        tts.set_sentence_embedding(prompt)
        tts.read_to_file(text_list=[sent], file_location=f"audios/{version}/Study/sent_prompt_{emotion}_{1}.flac", increased_compatibility_mode=True)

        emotion = "sadness"
        sent = "The sisters mourned as young hearts can mourn, and were especially grieved at the sight of their parents' sorrow."
        prompt = "Then she saw that her deliverance was near, and her heart leapt with joy."
        tts.set_sentence_embedding(prompt)
        tts.read_to_file(text_list=[sent], file_location=f"audios/{version}/Study/sent_prompt_{emotion}_{1}.flac", increased_compatibility_mode=True)

        emotion = "surprise"
        sent = "Really? I can't believe it! It's like a dream come true, I never expected that I would win The Nobel Prize!"
        prompt = "So the queen gave him the letter, and said that he might see for himself what was written in it."
        tts.set_sentence_embedding(prompt)
        tts.read_to_file(text_list=[sent], file_location=f"audios/{version}/Study/sent_prompt_{emotion}_{0}.flac", increased_compatibility_mode=True)

def test_study2_female(version, model_id="Meta", 
                      exec_device="cpu", 
                      speaker_reference=None, 
                      vocoder_model_path=None, 
                      biggan=False, 
                      sent_emb_extractor=None, 
                      word_emb_extractor=None, 
                      prompt:str=None, 
                      xvect_model=None, 
                      speaker_id=None):
    os.makedirs("audios", exist_ok=True)
    os.makedirs(f"audios/{version}", exist_ok=True)
    os.makedirs(f"audios/{version}/Study", exist_ok=True)
    tts = ToucanTTSInterface(device=exec_device, 
                             tts_model_path=model_id, 
                             vocoder_model_path=vocoder_model_path, 
                             faster_vocoder=not biggan, 
                             sent_emb_extractor=sent_emb_extractor, 
                             word_emb_extractor=word_emb_extractor, 
                             xvect_model=xvect_model)
    tts.set_language("en")
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if speaker_id is not None:
        tts.set_speaker_id(speaker_id)

        emotion = "anger"
        sent = "You can't be serious, how dare you not tell me you were going to marry her?"
        tts.read_to_file(text_list=[sent], file_location=f"audios/{version}/Study/sent_{emotion}_{0}.flac", increased_compatibility_mode=True)

        emotion = "joy"
        sent = "I really enjoy the beach in the summer."
        tts.read_to_file(text_list=[sent], file_location=f"audios/{version}/Study/sent_{emotion}_{0}.flac", increased_compatibility_mode=True)

        emotion = "neutral"
        sent = "You can go to the Employment Development Office and pick it up."
        tts.read_to_file(text_list=[sent], file_location=f"audios/{version}/Study/sent_{emotion}_{0}.flac", increased_compatibility_mode=True)

        emotion = "sadness"
        sent = "Lily broke up with me last week, in fact, she dumped me."
        tts.read_to_file(text_list=[sent], file_location=f"audios/{version}/Study/sent_{emotion}_{0}.flac", increased_compatibility_mode=True)

        emotion = "surprise"
        sent = "He was astonished when he saw them come alone, and asked what had happened to them."
        tts.read_to_file(text_list=[sent], file_location=f"audios/{version}/Study/sent_{emotion}_{1}.flac", increased_compatibility_mode=True)

def test_study2_female_prompt(version, model_id="Meta", 
                      exec_device="cpu", 
                      speaker_reference=None, 
                      vocoder_model_path=None, 
                      biggan=False, 
                      sent_emb_extractor=None, 
                      word_emb_extractor=None, 
                      prompt:str=None, 
                      xvect_model=None, 
                      speaker_id=None):
    os.makedirs("audios", exist_ok=True)
    os.makedirs(f"audios/{version}", exist_ok=True)
    os.makedirs(f"audios/{version}/Study", exist_ok=True)
    tts = ToucanTTSInterface(device=exec_device, 
                             tts_model_path=model_id, 
                             vocoder_model_path=vocoder_model_path, 
                             faster_vocoder=not biggan, 
                             sent_emb_extractor=sent_emb_extractor, 
                             word_emb_extractor=word_emb_extractor, 
                             xvect_model=xvect_model)
    tts.set_language("en")
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if speaker_id is not None:
        tts.set_speaker_id(speaker_id)

        emotion = "anger"
        sent = "You can't be serious, how dare you not tell me you were going to marry her?"
        prompt = "You can go to the Employment Development Office and pick it up."
        tts.set_sentence_embedding(prompt)
        tts.read_to_file(text_list=[sent], file_location=f"audios/{version}/Study/sent_prompt_{emotion}_{0}.flac", increased_compatibility_mode=True)

        emotion = "joy"
        sent = "I really enjoy the beach in the summer."
        prompt = "He was astonished when he saw them come alone, and asked what had happened to them."
        tts.set_sentence_embedding(prompt)
        tts.read_to_file(text_list=[sent], file_location=f"audios/{version}/Study/sent_prompt_{emotion}_{0}.flac", increased_compatibility_mode=True)

        emotion = "neutral"
        sent = "You can go to the Employment Development Office and pick it up."
        prompt = "I really enjoy the beach in the summer."
        tts.set_sentence_embedding(prompt)
        tts.read_to_file(text_list=[sent], file_location=f"audios/{version}/Study/sent_prompt_{emotion}_{0}.flac", increased_compatibility_mode=True)

        emotion = "sadness"
        sent = "Lily broke up with me last week, in fact, she dumped me."
        prompt = "You can't be serious, how dare you not tell me you were going to marry her?"
        tts.set_sentence_embedding(prompt)
        tts.read_to_file(text_list=[sent], file_location=f"audios/{version}/Study/sent_prompt_{emotion}_{0}.flac", increased_compatibility_mode=True)

        emotion = "surprise"
        sent = "He was astonished when he saw them come alone, and asked what had happened to them."
        prompt = "Lily broke up with me last week, in fact, she dumped me."
        tts.set_sentence_embedding(prompt)
        tts.read_to_file(text_list=[sent], file_location=f"audios/{version}/Study/sent_prompt_{emotion}_{1}.flac", increased_compatibility_mode=True)


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"0"
    exec_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"running on {exec_device}")

    use_speaker_reference = False
    use_sent_emb = True
    use_word_emb = False
    use_prompt = True
    use_xvect = False
    use_ecapa = False
    use_speaker_id = True

    if use_speaker_id:
        speaker_id = 4 + 1 + 24
    else:
        speaker_id = None

    if use_sent_emb:
        from Preprocessing.sentence_embeddings.EmotionRoBERTaSentenceEmbeddingExtractor import EmotionRoBERTaSentenceEmbeddingExtractor as SentenceEmbeddingExtractor
        sent_emb_extractor = SentenceEmbeddingExtractor(pooling="cls")
    else:
        sent_emb_extractor = None

    if use_word_emb:
        from Preprocessing.word_embeddings.EmotionRoBERTaWordEmbeddingExtractor import EmotionRoBERTaWordEmbeddingExtractor
        word_embedding_extractor = EmotionRoBERTaWordEmbeddingExtractor()
    else:
        word_embedding_extractor = None

    if use_speaker_reference:
        speaker_reference = "/mount/resources/speech/corpora/Emotional_Speech_Dataset_Singapore/0018/Surprise/0018_001431.wav"
    else:
        speaker_reference = None

    if use_prompt:
        #prompt = "I am so angry!"
        #prompt = "Roar with laughter, this is funny."
        #prompt = "Ew, this is disgusting."
        #prompt = "What a surprise!"
        #prompt = "This is very sad."
        #prompt = "I am so scared."
        #prompt = "I love that."
        #prompt = "He was furious."
        #prompt = "She didn't expect that."
        prompt = "That's ok."
        #prompt = "Oh, really?"
    else:
        prompt = None

    if use_xvect:
        from speechbrain.pretrained import EncoderClassifier
        xvect_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="./Models/Embedding/spkrec-xvect-voxceleb", run_opts={"device": exec_device})
    else:
        xvect_model = None

    if use_ecapa:
        from speechbrain.pretrained import EncoderClassifier
        ecapa_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="./Models/Embedding/spkrec-ecapa-voxceleb", run_opts={"device": exec_device})
    else:
        ecapa_model = None
    if ecapa_model is not None:
        xvect_model = ecapa_model

    test_sentence(version="ToucanTTS_Sent_Finetuning_2_80k",
                      model_id="Sent_Finetuning_2_80k",
                      exec_device=exec_device,
                      vocoder_model_path=None,
                      biggan=False,
                      speaker_reference=speaker_reference,
                      sent_emb_extractor=sent_emb_extractor,
                      word_emb_extractor=word_embedding_extractor,
                      prompt=prompt,
                      xvect_model=xvect_model, 
                      speaker_id=speaker_id)
