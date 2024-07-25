import glob
import json
import os
import random
import xml.etree.ElementTree as ET
from csv import DictReader
from pathlib import Path

import torch


# HELPER FUNCTIONS

def split_dictionary_into_chunks(input_dict, split_n):
    res = []
    new_dict = {}
    elements_per_dict = (len(input_dict.keys()) // split_n) + 1
    for k, v in input_dict.items():
        if len(new_dict) < elements_per_dict:
            new_dict[k] = v
        else:
            res.append(new_dict)
            new_dict = {k: v}
    res.append(new_dict)
    return res


def limit_to_n(path_to_transcript_dict, n=40000):
    # deprecated, we now just use the whole thing always, because there's a critical mass of data
    limited_dict = dict()
    if len(path_to_transcript_dict.keys()) > n:
        for key in random.sample(list(path_to_transcript_dict.keys()), n):
            limited_dict[key] = path_to_transcript_dict[key]
        return limited_dict
    else:
        return path_to_transcript_dict


def build_path_to_transcript_dict_multi_ling_librispeech_template(root):
    """
    https://arxiv.org/abs/2012.03411
    """
    path_to_transcript = dict()
    with open(os.path.join(root, "transcripts.txt"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            fields = line.split("\t")
            wav_folders = fields[0].split("_")
            wav_path = f"{root}/audio/{wav_folders[0]}/{wav_folders[1]}/{fields[0]}.flac"
            path_to_transcript[wav_path] = fields[1]
    return path_to_transcript


def build_path_to_transcript_dict_hui_template(root):
    """
    https://arxiv.org/abs/2106.06309
    """
    path_to_transcript = dict()
    for el in os.listdir(root):
        if os.path.isdir(os.path.join(root, el)):
            with open(os.path.join(root, el, "metadata.csv"), "r", encoding="utf8") as file:
                lookup = file.read()
            for line in lookup.split("\n"):
                if line.strip() != "":
                    norm_transcript = line.split("|")[1]
                    wav_path = os.path.join(root, el, "wavs", line.split("|")[0] + ".wav")
                    if os.path.exists(wav_path):
                        path_to_transcript[wav_path] = norm_transcript
    return path_to_transcript


def indic_voices_template(root, lang):
    path_to_transcript = dict()
    transcripts = list()
    import json
    for jpath in [f"{root}/{lang}/metadata_test.json",
                  f"{root}/{lang}/metadata_train.json"]:
        with open(jpath, encoding='utf-8', mode='r') as jfile:
            for line in jfile.read().split("\n"):
                if line.strip() != "":
                    transcripts.append(json.loads(line))
    for transcript in transcripts:
        path = f"{root}/{lang}/{lang}/wavs/{transcript['filepath']}"
        norm_text = transcript["normalized"]
        path_to_transcript[path] = norm_text
    return path_to_transcript


# ENGLISH

def build_path_to_transcript_dict_ears(re_cache=False):
    transcript_for_ears = {
        "emo_adoration_sentences"     : "You're just the sweetest person I know and I am so happy to call you my friend. I had the best time with you, I just adore you. I love this gift, thank you!",
        "emo_amazement_sentences"     : "I just love how you can play guitar. You're so impressive. I admire your abilities so much.",
        "emo_amusement_sentences"     : "The sound that baby just made was quite amusing. I liked that stand up comic, I found her pretty funny. What a fun little show to watch!",
        "emo_anger_sentences"         : "I'm so mad right now I could punch a hole in the wall. I can't believe he said that, he's such a jerk! There's a stop sign there and parents are just letting their kids run around!",
        "emo_confusion_sentences"     : "Huh, what is going on over here? What is this? Where are we going?",
        "emo_contentment_sentences"   : "I really enjoyed dinner tonight, it was quite nice. Everything is working out just fine. I'm good either way.",
        "emo_cuteness_sentences"      : "Look at that cute little kitty cat! Oh my goodness, she's so cute! That's the cutest thing I've ever seen!",
        "emo_desire_sentences"        : "Mmm that chocolate fudge lava cake looks divine. I want that car so badly. I can't wait to see you again.",
        "emo_disappointment_sentences": "I'm so disappointed in myself. I wish I had worked harder. I had such higher expectations for you. I really was hoping you were better than this.",
        "emo_disgust_sentences"       : "I have never seen anything grosser than this in my entire life. This is the worst dinner I've ever had. Yuck, I can't even look at that.",
        "emo_distress_sentences"      : "Oh god, I am not sure if we are going to make this flight on time. This is all too stressful to handle right now. I don't know where anything is and I'm running late.",
        "emo_embarassment_sentences"  : "I don't know what happened, I followed the recipe perfectly but the cake just deflated. I'm so embarrassed. I hope no one saw that, I'd be mortified if they did.",
        "emo_extasy_sentences"        : "This is the most exciting thing I've ever seen in my life! I can't believe I got to see that. I'm so excited, I've never been there before.",
        "emo_fear_sentences"          : "Did you hear that sound? I'm afraid someone or something is outside. Oh my gosh, what is that? What do you think is going to happen if we don't run?",
        "emo_guilt_sentences"         : "I'm sorry I did that to you. I really didn't mean to hurt you. I feel horrible that happened to you.",
        "emo_interest_sentences"      : "Hmm, I wonder what that cookie tastes like. Oh, what is that over there? So what exactly is it that you do?",
        "emo_neutral_sentences"       : "That wall in the living room is white. There is one more piece of bread in the pantry. The store closes at 8pm tonight.",
        "emo_pain_sentences"          : "Oh, this headache is the worst one I've ever had! My foot hurts so badly right now! I'm in terrible pain from that medication.",
        "emo_pride_sentences"         : "That was all me, I'm the one who found the project, created the company and made it succeed. I have worked hard to get here and I deserve it. I'm really proud of how well you did.",
        "emo_realization_sentences"   : "Wow, I never know that the body was made up of 75% water. Did you know that a flamingo is actually white but turns pink because it eats too many shrimp? Apparently dolphins sleep with one eye open.",
        "emo_relief_sentences"        : "I'm so relieved my taxes are done. That was so stressful. I'm so relieved that is over with. Thank goodness that's all done.",
        "emo_sadness_sentences"       : "I am so upset by the state of the world. I hope it gets better soon. I really miss her, life isn't the same without her. I'm sorry for your loss.",
        "emo_serenity_sentences"      : "This has been the most peaceful day of my life. I am very calm right now. I'm going to relax and take a nap here on the beach.",
        "rainbow_01_fast"             : "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors.",
        "rainbow_01_highpitch"        : "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors.",
        "rainbow_01_loud"             : "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors.",
        "rainbow_01_lowpitch"         : "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors.",
        "rainbow_01_regular"          : "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors.",
        "rainbow_01_slow"             : "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors.",
        "rainbow_01_whisper"          : "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors.",
        "rainbow_02_fast"             : "These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. There is, according to legend, a boiling pot of gold at one end.",
        "rainbow_02_highpitch"        : "These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. There is, according to legend, a boiling pot of gold at one end.",
        "rainbow_02_loud"             : "These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. There is, according to legend, a boiling pot of gold at one end.",
        "rainbow_02_lowpitch"         : "These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. There is, according to legend, a boiling pot of gold at one end.",
        "rainbow_02_regular"          : "These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. There is, according to legend, a boiling pot of gold at one end.",
        "rainbow_02_slow"             : "These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. There is, according to legend, a boiling pot of gold at one end.",
        "rainbow_02_whisper"          : "These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. There is, according to legend, a boiling pot of gold at one end.",
        "rainbow_03_fast"             : "People look, but no one ever finds it. When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow. Throughout the centuries people have explained the rainbow in various ways.",
        "rainbow_03_highpitch"        : "People look, but no one ever finds it. When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow. Throughout the centuries people have explained the rainbow in various ways.",
        "rainbow_03_loud"             : "People look, but no one ever finds it. When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow. Throughout the centuries people have explained the rainbow in various ways.",
        "rainbow_03_lowpitch"         : "People look, but no one ever finds it. When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow. Throughout the centuries people have explained the rainbow in various ways.",
        "rainbow_03_regular"          : "People look, but no one ever finds it. When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow. Throughout the centuries people have explained the rainbow in various ways.",
        "rainbow_03_slow"             : "People look, but no one ever finds it. When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow. Throughout the centuries people have explained the rainbow in various ways.",
        "rainbow_03_whisper"          : "People look, but no one ever finds it. When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow. Throughout the centuries people have explained the rainbow in various ways.",
        "rainbow_04_fast"             : "Some have accepted it as a miracle without physical explanation. To the Hebrews it was a token that there would be no more universal floods. The Greeks used to imagine that it was a sign from the gods to foretell war or heavy rain.",
        "rainbow_04_highpitch"        : "Some have accepted it as a miracle without physical explanation. To the Hebrews it was a token that there would be no more universal floods. The Greeks used to imagine that it was a sign from the gods to foretell war or heavy rain.",
        "rainbow_04_loud"             : "Some have accepted it as a miracle without physical explanation. To the Hebrews it was a token that there would be no more universal floods. The Greeks used to imagine that it was a sign from the gods to foretell war or heavy rain.",
        "rainbow_04_lowpitch"         : "Some have accepted it as a miracle without physical explanation. To the Hebrews it was a token that there would be no more universal floods. The Greeks used to imagine that it was a sign from the gods to foretell war or heavy rain.",
        "rainbow_04_regular"          : "Some have accepted it as a miracle without physical explanation. To the Hebrews it was a token that there would be no more universal floods. The Greeks used to imagine that it was a sign from the gods to foretell war or heavy rain.",
        "rainbow_04_slow"             : "Some have accepted it as a miracle without physical explanation. To the Hebrews it was a token that there would be no more universal floods. The Greeks used to imagine that it was a sign from the gods to foretell war or heavy rain.",
        "rainbow_04_whisper"          : "Some have accepted it as a miracle without physical explanation. To the Hebrews it was a token that there would be no more universal floods. The Greeks used to imagine that it was a sign from the gods to foretell war or heavy rain.",
        "rainbow_05_fast"             : "The Norsemen considered the rainbow as a bridge over which the gods passed from earth to their home in the sky. Others have tried to explain the phenomenon physically. Aristotle thought that the rainbow was caused by reflection of the sun's rays by the rain.",
        "rainbow_05_highpitch"        : "The Norsemen considered the rainbow as a bridge over which the gods passed from earth to their home in the sky. Others have tried to explain the phenomenon physically. Aristotle thought that the rainbow was caused by reflection of the sun's rays by the rain.",
        "rainbow_05_loud"             : "The Norsemen considered the rainbow as a bridge over which the gods passed from earth to their home in the sky. Others have tried to explain the phenomenon physically. Aristotle thought that the rainbow was caused by reflection of the sun's rays by the rain.",
        "rainbow_05_lowpitch"         : "The Norsemen considered the rainbow as a bridge over which the gods passed from earth to their home in the sky. Others have tried to explain the phenomenon physically. Aristotle thought that the rainbow was caused by reflection of the sun's rays by the rain.",
        "rainbow_05_regular"          : "The Norsemen considered the rainbow as a bridge over which the gods passed from earth to their home in the sky. Others have tried to explain the phenomenon physically. Aristotle thought that the rainbow was caused by reflection of the sun's rays by the rain.",
        "rainbow_05_slow"             : "The Norsemen considered the rainbow as a bridge over which the gods passed from earth to their home in the sky. Others have tried to explain the phenomenon physically. Aristotle thought that the rainbow was caused by reflection of the sun's rays by the rain.",
        "rainbow_05_whisper"          : "The Norsemen considered the rainbow as a bridge over which the gods passed from earth to their home in the sky. Others have tried to explain the phenomenon physically. Aristotle thought that the rainbow was caused by reflection of the sun's rays by the rain.",
        "rainbow_06_fast"             : "Since then physicists have found that it is not reflection, but refraction by the raindrops which causes the rainbows. Many complicated ideas about the rainbow have been formed.",
        "rainbow_06_highpitch"        : "Since then physicists have found that it is not reflection, but refraction by the raindrops which causes the rainbows. Many complicated ideas about the rainbow have been formed.",
        "rainbow_06_loud"             : "Since then physicists have found that it is not reflection, but refraction by the raindrops which causes the rainbows. Many complicated ideas about the rainbow have been formed.",
        "rainbow_06_lowpitch"         : "Since then physicists have found that it is not reflection, but refraction by the raindrops which causes the rainbows. Many complicated ideas about the rainbow have been formed.",
        "rainbow_06_regular"          : "Since then physicists have found that it is not reflection, but refraction by the raindrops which causes the rainbows. Many complicated ideas about the rainbow have been formed.",
        "rainbow_06_slow"             : "Since then physicists have found that it is not reflection, but refraction by the raindrops which causes the rainbows. Many complicated ideas about the rainbow have been formed.",
        "rainbow_06_whisper"          : "Since then physicists have found that it is not reflection, but refraction by the raindrops which causes the rainbows. Many complicated ideas about the rainbow have been formed.",
        "rainbow_07_fast"             : "The difference in the rainbow depends considerably upon the size of the drops, and the width of the colored band increases as the size of the drops increases. The actual primary rainbow observed is said to be the effect of super-imposition of a number of bows.",
        "rainbow_07_highpitch"        : "The difference in the rainbow depends considerably upon the size of the drops, and the width of the colored band increases as the size of the drops increases. The actual primary rainbow observed is said to be the effect of super-imposition of a number of bows.",
        "rainbow_07_loud"             : "The difference in the rainbow depends considerably upon the size of the drops, and the width of the colored band increases as the size of the drops increases. The actual primary rainbow observed is said to be the effect of super-imposition of a number of bows.",
        "rainbow_07_lowpitch"         : "The difference in the rainbow depends considerably upon the size of the drops, and the width of the colored band increases as the size of the drops increases. The actual primary rainbow observed is said to be the effect of super-imposition of a number of bows.",
        "rainbow_07_regular"          : "The difference in the rainbow depends considerably upon the size of the drops, and the width of the colored band increases as the size of the drops increases. The actual primary rainbow observed is said to be the effect of super-imposition of a number of bows.",
        "rainbow_07_slow"             : "The difference in the rainbow depends considerably upon the size of the drops, and the width of the colored band increases as the size of the drops increases. The actual primary rainbow observed is said to be the effect of super-imposition of a number of bows.",
        "rainbow_07_whisper"          : "The difference in the rainbow depends considerably upon the size of the drops, and the width of the colored band increases as the size of the drops increases. The actual primary rainbow observed is said to be the effect of super-imposition of a number of bows.",
        "rainbow_08_fast"             : "If the red of the second bow falls upon the green of the first, the result is to give a bow with an abnormally wide yellow band, since red and green light when mixed form yellow. This is a very common type of bow, one showing mainly red and yellow, with little or no green or blue.",
        "rainbow_08_highpitch"        : "If the red of the second bow falls upon the green of the first, the result is to give a bow with an abnormally wide yellow band, since red and green light when mixed form yellow. This is a very common type of bow, one showing mainly red and yellow, with little or no green or blue.",
        "rainbow_08_loud"             : "If the red of the second bow falls upon the green of the first, the result is to give a bow with an abnormally wide yellow band, since red and green light when mixed form yellow. This is a very common type of bow, one showing mainly red and yellow, with little or no green or blue.",
        "rainbow_08_lowpitch"         : "If the red of the second bow falls upon the green of the first, the result is to give a bow with an abnormally wide yellow band, since red and green light when mixed form yellow. This is a very common type of bow, one showing mainly red and yellow, with little or no green or blue.",
        "rainbow_08_regular"          : "If the red of the second bow falls upon the green of the first, the result is to give a bow with an abnormally wide yellow band, since red and green light when mixed form yellow. This is a very common type of bow, one showing mainly red and yellow, with little or no green or blue.",
        "rainbow_08_slow"             : "If the red of the second bow falls upon the green of the first, the result is to give a bow with an abnormally wide yellow band, since red and green light when mixed form yellow. This is a very common type of bow, one showing mainly red and yellow, with little or no green or blue.",
        "rainbow_08_whisper"          : "If the red of the second bow falls upon the green of the first, the result is to give a bow with an abnormally wide yellow band, since red and green light when mixed form yellow. This is a very common type of bow, one showing mainly red and yellow, with little or no green or blue.",
        "sentences_01_fast"           : "I will not stay here. God, we simply must dress the character. Stay, stay, I will go myself. May one ask what it is for. He rushed to the window and opened the movable pane.",
        "sentences_01_highpitch"      : "I will not stay here. God, we simply must dress the character. Stay, stay, I will go myself. May one ask what it is for. He rushed to the window and opened the movable pane.",
        "sentences_01_loud"           : "I will not stay here. God, we simply must dress the character. Stay, stay, I will go myself. May one ask what it is for. He rushed to the window and opened the movable pane.",
        "sentences_01_lowpitch"       : "I will not stay here. God, we simply must dress the character. Stay, stay, I will go myself. May one ask what it is for. He rushed to the window and opened the movable pane.",
        "sentences_01_regular"        : "I will not stay here. God, we simply must dress the character. Stay, stay, I will go myself. May one ask what it is for. He rushed to the window and opened the movable pane.",
        "sentences_01_slow"           : "I will not stay here. God, we simply must dress the character. Stay, stay, I will go myself. May one ask what it is for. He rushed to the window and opened the movable pane.",
        "sentences_01_whisper"        : "I will not stay here. God, we simply must dress the character. Stay, stay, I will go myself. May one ask what it is for. He rushed to the window and opened the movable pane.",
        "sentences_02_fast"           : "It might happen, he added with an involuntary smile. It is sold, sir, was again his laconic reply. And you must have some water, my dear fellow. What is that flying about? Who wants a dead cert for the Gold cup?",
        "sentences_02_highpitch"      : "It might happen, he added with an involuntary smile. It is sold, sir, was again his laconic reply. And you must have some water, my dear fellow. What is that flying about? Who wants a dead cert for the Gold cup?",
        "sentences_02_loud"           : "It might happen, he added with an involuntary smile. It is sold, sir, was again his laconic reply. And you must have some water, my dear fellow. What is that flying about? Who wants a dead cert for the Gold cup?",
        "sentences_02_lowpitch"       : "It might happen, he added with an involuntary smile. It is sold, sir, was again his laconic reply. And you must have some water, my dear fellow. What is that flying about? Who wants a dead cert for the Gold cup?",
        "sentences_02_regular"        : "It might happen, he added with an involuntary smile. It is sold, sir, was again his laconic reply. And you must have some water, my dear fellow. What is that flying about? Who wants a dead cert for the Gold cup?",
        "sentences_02_slow"           : "It might happen, he added with an involuntary smile. It is sold, sir, was again his laconic reply. And you must have some water, my dear fellow. What is that flying about? Who wants a dead cert for the Gold cup?",
        "sentences_02_whisper"        : "It might happen, he added with an involuntary smile. It is sold, sir, was again his laconic reply. And you must have some water, my dear fellow. What is that flying about? Who wants a dead cert for the Gold cup?",
        "sentences_03_whisper"        : "Had it been but one, it had been easy. We have boxed the compass among us. I shall rush out and prevent it. All that is mean slander. The doctor seemed tired and in a hurry.",
        "sentences_04_whisper"        : "I only heard it last night. We had now got into the month of March. But go thy ways; I had forgot. Conceited fellow with his waxed up moustache! Anne's unhappiness continued for a week.",
        "sentences_05_loud"           : "In fact, the count's face brightened. For God's sake, talk to her. In what an amiable light does this place him! Take me out of my way. I heard many things in hell.",
        "sentences_06_loud"           : "Yes; but we do not invite people of fashion. You see what he writes. Silent with awe and pity I went to her bedside. Happy to say, I never knew him. Birthdays are of no importance to a rational being.",
        "sentences_07_slow"           : "But it may all be put in two words. Clear up the room, the sick man said with effort. He was still in sight. He delayed; he seemed almost afraid of something. Then they carried me in.",
        "sentences_08_slow"           : "But I have never been presented. But we were only in fun! Now, look at that third name. And serve them both right, too. Good glass of burgundy take away that.",
        "sentences_09_fast"           : "And it seemed to her that God heard her prayer. My word, I admire you. I also have a pious visit to pay. She has promised to come on the twentieth. I want to tell you something.",
        "sentences_10_fast"           : "Oh, sir, it will break bones. I am very glad to see you. This question absorbed all his mental powers. Before going away forever, I'll tell him all. I told you it was mother.",
        "sentences_11_highpitch"      : "You're all in good spirits. They might retreat and leave the pickets. But I like sentimental people. Our potato crop is very good this year. Why is the chestnut on the right?",
        "sentences_12_highpitch"      : "His room was on the first floor. I have had a pattern in my hand. The knocking still continued and grew louder. May my sorrows ever shun the light. How must I arrange it, then?",
        "sentences_13_lowpitch"       : "Just read it out to me. I shall take your advice in every particular. What mortal imagination could conceive it? The gate was again hidden by smoke. After a while I left him.",
        "sentences_14_lowpitch"       : "There was a catch in her breath. They told me, but I didn't understand. What a cabin it is. A cry of joy broke from his lips. He had obviously prepared the sentence beforehand.",
        "sentences_15_regular"        : "They were all sitting in her room. So that's how it stands. He did not know why he embraced it. Why don't you speak, cousin? I didn't tell a tale.",
        "sentences_16_regular"        : "My head aches dreadfully now. Not to say every word. I have only found out. He is trying to discover something. I have done my duty.",
        "sentences_17_regular"        : "I always had a value for him. He is a deceiver and a villain. But those tears were pleasant to them both. She conquered her fears, and spoke. Oh, he couldn't overhear me at the door.",
        "sentences_18_regular"        : "How could I have said it more directly? She remembered her oath. My kingdom for a drink! Have they caught the little girl and the boy? Then she gave him the dry bread.",
        "sentences_19_regular"        : "Your sister is given to government. Water was being sprinkled on his face. The clumsy things are dear. He jumped up and sat on the sofa. How do you know her?",
        "sentences_20_regular"        : "I never could guess a riddle in my life. The expression of her face was cold. Besides, what on earth could happen to you? Allow me to give you a piece of advice. This must be stopped at once.",
        "sentences_21_regular"        : "The lawyer was right about that. You are fond of fighting. Every word is so deep. So you were never in London before? Death is now, perhaps, striking a fourth blow.",
        "sentences_22_regular"        : "It seemed that sleep and night had resumed their empire. The snowstorm was still raging. But we'll talk later on. Take the baby, Mum, and give me your book. The doctor gave him his hand.",
        "sentences_23_regular"        : "It is, nevertheless, conclusive to my mind. Give this to the countess. It is only a question of a few hours. No, we don't keep a cat. The cool evening air refreshed him.",
        "sentences_24_regular"        : "You can well enjoy the evening now. We'll make up for it now. The weakness of a murderer. But they wouldn't leave me alone. The telegram was from his wife."
    }
    root = "/mount/resources/speech/corpora/EARS/"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        for speaker in os.listdir(root):
            if os.path.isdir(os.path.join(root, speaker)):
                for sentence_type in transcript_for_ears:
                    path = os.path.join(root, speaker, sentence_type + ".wav")
                    path_to_transcript[path] = transcript_for_ears[sentence_type]
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_mls_english(re_cache=False):
    lang = "english"
    root = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train"
    cache_path = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train/pttd_cache.pt"
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = build_path_to_transcript_dict_multi_ling_librispeech_template(root=root)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_gigaspeech(re_cache=False):
    root = "/mount/resources/speech/corpora/GigaSpeech/"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(os.path.join(root, "transcripts_only_clean_samples.txt"), "r", encoding="utf8") as file:
            lookup = file.read()
        for line in lookup.split("\n"):
            if line.strip() != "":
                fields = line.split("\t")
                norm_transcript = fields[1]
                wav_path = fields[0]
                path_to_transcript[wav_path] = norm_transcript
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_elizabeth(re_cache=False):
    root = "/mount/resources/speech/corpora/MAILabs_british_single_speaker_elizabeth"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        for el in os.listdir(root):
            if os.path.isdir(os.path.join(root, el)):
                with open(os.path.join(root, el, "metadata.csv"), "r", encoding="utf8") as file:
                    lookup = file.read()
                for line in lookup.split("\n"):
                    if line.strip() != "":
                        norm_transcript = line.split("|")[2]
                        wav_path = os.path.join(root, el, "wavs", line.split("|")[0] + ".wav")
                        if os.path.exists(wav_path):
                            path_to_transcript[wav_path] = norm_transcript
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_nancy(re_cache=False):
    root = "/mount/resources/speech/corpora/NancyKrebs"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(os.path.join(root, "metadata.csv"), "r", encoding="utf8") as file:
            lookup = file.read()
        for line in lookup.split("\n"):
            if line.strip() != "":
                norm_transcript = line.split("|")[1]
                wav_path = os.path.join(root, "wav", line.split("|")[0] + ".wav")
                if os.path.exists(wav_path):
                    path_to_transcript[wav_path] = norm_transcript
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_integration_test(re_cache=True):
    root = "/mount/resources/speech/corpora/NancyKrebs"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(os.path.join(root, "metadata.csv"), "r", encoding="utf8") as file:
            lookup = file.read()
        for line in lookup.split("\n")[:500]:
            if line.strip() != "":
                norm_transcript = line.split("|")[1]
                wav_path = os.path.join(root, "wav", line.split("|")[0] + ".wav")
                if os.path.exists(wav_path):
                    path_to_transcript[wav_path] = norm_transcript
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_CREMA_D(re_cache=False):
    root = "/mount/resources/speech/corpora/CREMA_D/"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        identifier_to_sent = {"IEO": "It's eleven o'clock.",
                              "TIE": "That is exactly what happened.",
                              "IOM": "I'm on my way to the meeting.",
                              "IWW": "I wonder what this is about.",
                              "TAI": "The airplane is almost full.",
                              "MTI": "Maybe tomorrow it will be cold.",
                              "IWL": "I would like a new alarm clock.",
                              "ITH": "I think, I have a doctor's appointment.",
                              "DFA": "Don't forget a jacket.",
                              "ITS": "I think, I've seen this before.",
                              "TSI": "The surface is slick.",
                              "WSI": "We'll stop in a couple of minutes."}
        path_to_transcript = dict()
        for file in os.listdir(root):
            if file.endswith(".wav"):
                path_to_transcript[root + file] = identifier_to_sent[file.split("_")[1]]
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_EmoV_DB(re_cache=False):
    root = "/mount/resources/speech/corpora/EmoV_DB/"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(os.path.join(root, "labels.txt"), "r", encoding="utf8") as file:
            lookup = file.read()
        identifier_to_sent = dict()
        for line in lookup.split("\n"):
            if line.strip() != "":
                identifier_to_sent[line.split()[0]] = " ".join(line.split()[1:])
        for file in os.listdir(root):
            if file.endswith(".wav"):
                path_to_transcript[root + file] = identifier_to_sent[file[-14:-10]]
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_ryanspeech(re_cache=False):
    root = "/mount/resources/speech/corpora/RyanSpeech"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript_dict = dict()
        with open(root + "/metadata.csv", mode="r", encoding="utf8") as f:
            transcripts = f.read().split("\n")
        for transcript in transcripts:
            if transcript.strip() != "":
                parsed_line = transcript.split("|")
                audio_file = f"{root}/wavs/{parsed_line[0]}.wav"
                path_to_transcript_dict[audio_file] = parsed_line[2]
        torch.save(path_to_transcript_dict, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_RAVDESS(re_cache=False):
    root = "/mount/resources/speech/corpora/RAVDESS"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript_dict = dict()
        for speaker_dir in os.listdir(root):
            for audio_file in os.listdir(os.path.join(root, speaker_dir)):
                if audio_file.split("-")[4] == "01":
                    path_to_transcript_dict[os.path.join(root, speaker_dir, audio_file)] = "Kids are talking by the door."
                else:
                    path_to_transcript_dict[os.path.join(root, speaker_dir, audio_file)] = "Dogs are sitting by the door."
        torch.save(path_to_transcript_dict, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_ESDS(re_cache=False):
    root = "/mount/resources/speech/corpora/Emotional_Speech_Dataset_Singapore"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript_dict = dict()
        for speaker_dir in os.listdir(root):
            if speaker_dir.startswith("00"):
                if int(speaker_dir) > 10:
                    with open(f"{root}/{speaker_dir}/fixed_unicode.txt", mode="r", encoding="utf8") as f:
                        transcripts = f.read()
                    for line in transcripts.replace("\n\n", "\n").replace(",", ", ").split("\n"):
                        if line.strip() != "":
                            filename, text, emo_dir = line.split("\t")
                            filename = speaker_dir + "_" + filename.split("_")[1]
                            path_to_transcript_dict[f"{root}/{speaker_dir}/{emo_dir}/{filename}.wav"] = text
        torch.save(path_to_transcript_dict, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_nvidia_hifitts(re_cache=False):
    root = "/mount/resources/speech/corpora/hi_fi_tts_v0"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        transcripts = list()
        import json
        for jpath in [f"{root}/6097_manifest_clean_dev.json",
                      f"{root}/6097_manifest_clean_test.json",
                      f"{root}/6097_manifest_clean_train.json",
                      f"{root}/9017_manifest_clean_dev.json",
                      f"{root}/9017_manifest_clean_test.json",
                      f"{root}/9017_manifest_clean_train.json",
                      f"{root}/92_manifest_clean_dev.json",
                      f"{root}/92_manifest_clean_test.json",
                      f"{root}/92_manifest_clean_train.json"]:
            with open(jpath, encoding='utf-8', mode='r') as jfile:
                for line in jfile.read().split("\n"):
                    if line.strip() != "":
                        transcripts.append(json.loads(line))
        for transcript in transcripts:
            path = transcript["audio_filepath"]
            norm_text = transcript["text_normalized"]
            path_to_transcript[f"{root}/{path}"] = norm_text
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_blizzard_2013(re_cache=False):
    root = "/mount/resources/speech/corpora/Blizzard2013/train/segmented/"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(root + "prompts.gui", encoding="utf8") as f:
            transcriptions = f.read()
        blocks = transcriptions.split("||\n")
        for block in blocks:
            trans_lines = block.split("\n")
            if trans_lines[0].strip() != "":
                transcript = trans_lines[1].replace("@", "").replace("#", ",").replace("|", "").replace(";", ",").replace(
                    ":", ",").replace(" 's", "'s").replace(", ,", ",").replace("  ", " ").replace(" ,", ",").replace(" .",
                                                                                                                     ".").replace(
                    " ?", "?").replace(" !", "!").rstrip(" ,")
                path_to_transcript[root + "wavn/" + trans_lines[0] + ".wav"] = transcript
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_vctk(re_cache=False):
    root = "/mount/resources/speech/corpora/VCTK"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        for transcript_dir in os.listdir("/mount/resources/speech/corpora/VCTK/txt"):
            for transcript_file in os.listdir(f"/mount/resources/speech/corpora/VCTK/txt/{transcript_dir}"):
                if transcript_file.endswith(".txt"):
                    with open(f"/mount/resources/speech/corpora/VCTK/txt/{transcript_dir}/" + transcript_file, 'r',
                              encoding='utf8') as tf:
                        transcript = tf.read()
                    wav_path = f"/mount/resources/speech/corpora/VCTK/wav48_silence_trimmed/{transcript_dir}/" + transcript_file.rstrip(
                        ".txt") + "_mic2.flac"
                    if os.path.exists(wav_path):
                        path_to_transcript[wav_path] = transcript
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_libritts_all_clean(re_cache=False):
    root = "/mount/resources/speech/corpora/LibriTTS_R/"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_train = "/mount/resources/speech/corpora/LibriTTS_R/"  # using all files from the "clean" subsets from LibriTTS-R https://arxiv.org/abs/2305.18802
        path_to_transcript = dict()
        for speaker in os.listdir(path_train):
            for chapter in os.listdir(os.path.join(path_train, speaker)):
                for file in os.listdir(os.path.join(path_train, speaker, chapter)):
                    if file.endswith("normalized.txt"):
                        with open(os.path.join(path_train, speaker, chapter, file), 'r', encoding='utf8') as tf:
                            transcript = tf.read()
                        wav_file = file.split(".")[0] + ".wav"
                        path_to_transcript[os.path.join(path_train, speaker, chapter, wav_file)] = transcript
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_libritts_other500(re_cache=False):
    root = "/mount/resources/asr-data/LibriTTS/train-other-500"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_train = "/mount/resources/asr-data/LibriTTS/train-other-500"
        path_to_transcript = dict()
        for speaker in os.listdir(path_train):
            for chapter in os.listdir(os.path.join(path_train, speaker)):
                for file in os.listdir(os.path.join(path_train, speaker, chapter)):
                    if file.endswith("normalized.txt"):
                        with open(os.path.join(path_train, speaker, chapter, file), 'r', encoding='utf8') as tf:
                            transcript = tf.read()
                        wav_file = file.split(".")[0] + ".wav"
                        path_to_transcript[os.path.join(path_train, speaker, chapter, wav_file)] = transcript
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_ljspeech(re_cache=False):
    root = "/mount/resources/speech/corpora/LJSpeech/"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        for transcript_file in os.listdir("/mount/resources/speech/corpora/LJSpeech/16kHz/txt"):
            with open("/mount/resources/speech/corpora/LJSpeech/16kHz/txt/" + transcript_file, 'r', encoding='utf8') as tf:
                transcript = tf.read()
            wav_path = "/mount/resources/speech/corpora/LJSpeech/16kHz/wav/" + transcript_file.rstrip(".txt") + ".wav"
            path_to_transcript[wav_path] = transcript
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_jenny(re_cache=False):
    """
    https://www.kaggle.com/datasets/noml4u/jenny-tts-dataset
    https://github.com/dioco-group/jenny-tts-dataset

    Dataset of Speaker Jenny (Dioco) with an Irish accent
    """
    root = "/mount/resources/speech/corpora/Jenny/"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open("/mount/resources/speech/corpora/Jenny/metadata.csv", encoding="utf8") as f:
            transcriptions = f.read()
        trans_lines = transcriptions.split("\n")
        for line in trans_lines:
            if line.strip() != "":
                path_to_transcript["/mount/resources/speech/corpora/Jenny/" + line.split("|")[0] + "_silence.flac"] = line.split("|")[1]
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


# GERMAN

def build_path_to_transcript_dict_mls_german(re_cache=False):
    lang = "german"
    root = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = build_path_to_transcript_dict_multi_ling_librispeech_template(root=root)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_karlsson(re_cache=False):
    root = "/mount/resources/speech/corpora/HUI_German/Karlsson"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = build_path_to_transcript_dict_hui_template(root=root)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_eva(re_cache=False):
    root = "/mount/resources/speech/corpora/HUI_German/Eva"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = build_path_to_transcript_dict_hui_template(root=root)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_bernd(re_cache=False):
    root = "/mount/resources/speech/corpora/HUI_German/Bernd"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = build_path_to_transcript_dict_hui_template(root=root)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_friedrich(re_cache=False):
    root = "/mount/resources/speech/corpora/HUI_German/Friedrich"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = build_path_to_transcript_dict_hui_template(root=root)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_hokus(re_cache=False):
    root = "/mount/resources/speech/corpora/HUI_German/Hokus"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = build_path_to_transcript_dict_hui_template(root=root)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_hui_others(re_cache=False):
    root = "/mount/resources/speech/corpora/HUI_German/others"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        pttd = dict()
        for speaker in os.listdir(root):
            pttd.update(build_path_to_transcript_dict_hui_template(root=f"{root}/{speaker}"))
        torch.save(pttd, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_thorsten_neutral(re_cache=False):
    root = "/mount/resources/speech/corpora/ThorstenDatasets/thorsten-de_v03"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(root + "/metadata_train.csv", encoding="utf8") as f:
            transcriptions = f.read()
        with open(root + "/metadata_val.csv", encoding="utf8") as f:
            transcriptions += "\n" + f.read()
        trans_lines = transcriptions.split("\n")
        for line in trans_lines:
            if line.strip() != "":
                path_to_transcript[root + "/wavs/" + line.split("|")[0] + ".wav"] = \
                    line.split("|")[1]
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_thorsten_2022_10(re_cache=False):
    root = "/mount/resources/speech/corpora/ThorstenDatasets/ThorstenVoice-Dataset_2022.10"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(root + "/metadata_train.csv", encoding="utf8") as f:
            transcriptions = f.read()
        with open(root + "/metadata_dev.csv", encoding="utf8") as f:
            transcriptions += "\n" + f.read()
        with open(root + "/metadata_test.csv", encoding="utf8") as f:
            transcriptions += "\n" + f.read()
        trans_lines = transcriptions.split("\n")
        for line in trans_lines:
            if line.strip() != "":
                path_to_transcript[root + "/wavs/" + line.split("|")[0] + ".wav"] = \
                    line.split("|")[1]
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_thorsten_emotional(re_cache=False):
    root = "/mount/resources/speech/corpora/ThorstenDatasets/thorsten-emotional_v02"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(root + "/thorsten-emotional-metadata.csv", encoding="utf8") as f:
            transcriptions = f.read()
        trans_lines = transcriptions.split("\n")
        for line in trans_lines:
            if line.strip() != "":
                path_to_transcript[root + "/amused/" + line.split("|")[0] + ".wav"] = line.split("|")[1]
                path_to_transcript[root + "/angry/" + line.split("|")[0] + ".wav"] = line.split("|")[1]
                path_to_transcript[root + "/disgusted/" + line.split("|")[0] + ".wav"] = line.split("|")[1]
                path_to_transcript[root + "/neutral/" + line.split("|")[0] + ".wav"] = line.split("|")[1]
                path_to_transcript[root + "/sleepy/" + line.split("|")[0] + ".wav"] = line.split("|")[1]
                path_to_transcript[root + "/surprised/" + line.split("|")[0] + ".wav"] = line.split("|")[1]
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


# FRENCH

def build_path_to_transcript_dict_mls_french(re_cache=False):
    lang = "french"
    root = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = build_path_to_transcript_dict_multi_ling_librispeech_template(root=root)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_blizzard2023_ad_silence_removed(re_cache=False):
    root = "/mount/resources/speech/corpora/Blizzard2023/AD_silence_removed"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(os.path.join(root, "transcript.tsv"), "r", encoding="utf8") as file:
            lookup = file.read()
        for line in lookup.split("\n"):
            if line.strip() != "":
                norm_transcript = line.split("\t")[1]
                wav_path = os.path.join(root, line.split("\t")[0].split("/")[-1])
                if os.path.exists(wav_path):
                    path_to_transcript[wav_path] = norm_transcript.replace("§", "").replace("#", "").replace("~", "").replace(" »", '"').replace("« ", '"').replace("»", '"').replace("«", '"')
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_blizzard2023_neb_silence_removed(re_cache=False):
    root = "/mount/resources/speech/corpora/Blizzard2023/NEB_silence_removed"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(os.path.join(root, "transcript.tsv"), "r", encoding="utf8") as file:
            lookup = file.read()
        for line in lookup.split("\n"):
            if line.strip() != "":
                norm_transcript = line.split("\t")[1]
                wav_path = os.path.join(root, line.split("\t")[0].split("/")[-1])
                if os.path.exists(wav_path):
                    path_to_transcript[wav_path] = norm_transcript.replace("§", "").replace("#", "").replace("~", "").replace(" »", '"').replace("« ", '"').replace("»", '"').replace("«", '"')
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_blizzard2023_neb_e_silence_removed(re_cache=False):
    root = "/mount/resources/speech/corpora/Blizzard2023/enhanced_NEB_subset_silence_removed"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(os.path.join(root, "transcript.tsv"), "r", encoding="utf8") as file:
            lookup = file.read()
        for line in lookup.split("\n"):
            if line.strip() != "":
                norm_transcript = line.split("\t")[1]
                wav_path = os.path.join(root, line.split("\t")[0].split("/")[-1])
                if os.path.exists(wav_path):
                    path_to_transcript[wav_path] = norm_transcript.replace("§", "").replace("#", "").replace("~", "").replace(" »", '"').replace("« ", '"').replace("»", '"').replace("«", '"')
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_synpaflex_norm_subset(re_cache=False):
    """
    Contributed by https://github.com/tomschelsen
    """
    root = "/mount/resources/speech/corpora/synpaflex-corpus/5/v0.1/"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        for text_path in glob.iglob(os.path.join(root, "**/*_norm.txt"), recursive=True):
            with open(text_path, "r", encoding="utf8") as file:
                norm_transcript = file.read()
            path_obj = Path(text_path)
            wav_path = str((path_obj.parent.parent / path_obj.name[:-9]).with_suffix(".wav"))
            if Path(wav_path).exists():
                path_to_transcript[wav_path] = norm_transcript
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_siwis_subset(re_cache=False):
    """
    Contributed by https://github.com/tomschelsen
    """
    root = "/mount/resources/speech/corpora/SiwisFrenchSpeechSynthesisDatabase/"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        # part4 and part5 are not segmented
        sub_dirs = ["part1", "part2", "part3"]
        path_to_transcript = dict()
        for sd in sub_dirs:
            for text_path in glob.iglob(os.path.join(root, "text", sd, "*.txt")):
                with open(text_path, "r", encoding="utf8") as file:
                    norm_transcript = file.read()
                path_obj = Path(text_path)
                wav_path = str((path_obj.parent.parent.parent / "wavs" / sd / path_obj.stem).with_suffix(".wav"))
                if Path(wav_path).exists():
                    path_to_transcript[wav_path] = norm_transcript
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_css10fr(re_cache=False):
    language = "french"
    root = f"/mount/resources/speech/corpora/CSS10/{language}"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(f"/mount/resources/speech/corpora/CSS10/{language}/transcript.txt", encoding="utf8") as f:
            transcriptions = f.read()
        trans_lines = transcriptions.split("\n")
        for line in trans_lines:
            if line.strip() != "":
                path_to_transcript[f"/mount/resources/speech/corpora/CSS10/{language}/{line.split('|')[0]}"] = \
                    line.split("|")[2]
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


# SPANISH

def build_path_to_transcript_dict_mls_spanish(re_cache=False):
    lang = "spanish"
    root = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = build_path_to_transcript_dict_multi_ling_librispeech_template(root=root)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_css10es(re_cache=False):
    language = "spanish"
    root = f"/mount/resources/speech/corpora/CSS10/{language}"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(f"/mount/resources/speech/corpora/CSS10/{language}/transcript.txt", encoding="utf8") as f:
            transcriptions = f.read()
        trans_lines = transcriptions.split("\n")
        for line in trans_lines:
            if line.strip() != "":
                path_to_transcript[f"/mount/resources/speech/corpora/CSS10/{language}/{line.split('|')[0]}"] = \
                    line.split("|")[2]
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_spanish_blizzard_train(re_cache=False):
    root = "/mount/resources/speech/corpora/Blizzard2021/spanish_blizzard_release_2021_v2/hub"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(os.path.join(root, "train_text.txt"), "r", encoding="utf8") as file:
            lookup = file.read()
        for line in lookup.split("\n"):
            if line.strip() != "":
                norm_transcript = line.split("\t")[1]
                wav_path = os.path.join(root, "train_wav", line.split("\t")[0] + ".wav")
                if os.path.exists(wav_path):
                    path_to_transcript[wav_path] = norm_transcript
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


# PORTUGUESE

def build_path_to_transcript_dict_mls_portuguese(re_cache=False):
    lang = "portuguese"
    root = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = build_path_to_transcript_dict_multi_ling_librispeech_template(root=root)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


# POLISH

def build_path_to_transcript_dict_mls_polish(re_cache=False):
    lang = "polish"
    root = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = build_path_to_transcript_dict_multi_ling_librispeech_template(root=root)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


# ITALIAN

def build_path_to_transcript_dict_mls_italian(re_cache=False):
    lang = "italian"
    root = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = build_path_to_transcript_dict_multi_ling_librispeech_template(root=root)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


# DUTCH

def build_path_to_transcript_dict_mls_dutch(re_cache=False):
    lang = "dutch"
    root = f"/mount/resources/speech/corpora/MultiLingLibriSpeech/mls_{lang}/train"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = build_path_to_transcript_dict_multi_ling_librispeech_template(root=root)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_css10nl(re_cache=False):
    language = "dutch"
    root = f"/mount/resources/speech/corpora/CSS10/{language}"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(f"/mount/resources/speech/corpora/CSS10/{language}/transcript.txt", encoding="utf8") as f:
            transcriptions = f.read()
        trans_lines = transcriptions.split("\n")
        for line in trans_lines:
            if line.strip() != "":
                path_to_transcript[f"/mount/resources/speech/corpora/CSS10/{language}/{line.split('|')[0]}"] = \
                    line.split("|")[2]
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


# GREEK

def build_path_to_transcript_dict_css10el(re_cache=False):
    language = "greek"
    root = f"/mount/resources/speech/corpora/CSS10/{language}"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(f"/mount/resources/speech/corpora/CSS10/{language}/transcript.txt", encoding="utf8") as f:
            transcriptions = f.read()
        trans_lines = transcriptions.split("\n")
        for line in trans_lines:
            if line.strip() != "":
                path_to_transcript[f"/mount/resources/speech/corpora/CSS10/{language}/{line.split('|')[0]}"] = \
                    line.split("|")[2]
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


# FINNISH

def build_path_to_transcript_dict_css10fi(re_cache=False):
    language = "finnish"
    root = f"/mount/resources/speech/corpora/CSS10/{language}"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(f"/mount/resources/speech/corpora/CSS10/{language}/transcript.txt", encoding="utf8") as f:
            transcriptions = f.read()
        trans_lines = transcriptions.split("\n")
        for line in trans_lines:
            if line.strip() != "":
                path_to_transcript[f"/mount/resources/speech/corpora/CSS10/{language}/{line.split('|')[0]}"] = \
                    line.split("|")[2]
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


# VIETNAMESE


def build_path_to_transcript_dict_VIVOS_viet(re_cache=False):
    root = "/mount/resources/speech/corpora/VIVOS_vietnamese/train"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript_dict = dict()
        with open(root + "/prompts.txt", mode="r", encoding="utf8") as f:
            transcripts = f.read().split("\n")
        for transcript in transcripts:
            if transcript.strip() != "":
                parsed_line = transcript.split(" ")
                audio_file = f"{root}/waves/{parsed_line[0][:10]}/{parsed_line[0]}.wav"
                path_to_transcript_dict[audio_file] = " ".join(parsed_line[1:]).lower()
        torch.save(path_to_transcript_dict, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_vietTTS(re_cache=False):
    root = "/mount/resources/speech/corpora/VietTTS"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(root + "/meta_data.tsv", encoding="utf8") as f:
            transcriptions = f.read()
        for line in transcriptions.split("\n"):
            if line.strip() != "":
                parsed_line = line.split(".wav")
                audio_path = parsed_line[0]
                transcript = parsed_line[1]
                path_to_transcript[os.path.join(root, audio_path + ".wav")] = transcript.strip()
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


# CHINESE

def build_path_to_transcript_dict_aishell3(re_cache=False):
    root = "/mount/resources/speech/corpora/aishell3/train"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript_dict = dict()
        with open(root + "/label_train-set.txt", mode="r", encoding="utf8") as f:
            transcripts = f.read().replace("$", "").replace("%", " ").split("\n")
        for transcript in transcripts:
            if transcript.strip() != "" and not transcript.startswith("#"):
                parsed_line = transcript.split("|")
                audio_file = f"{root}/wav/{parsed_line[0][:7]}/{parsed_line[0]}.wav"
                kanji = parsed_line[2]
                path_to_transcript_dict[audio_file] = kanji
        torch.save(path_to_transcript_dict, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_css10cmn(re_cache=False):
    language = "chinese"
    root = f"/mount/resources/speech/corpora/CSS10/{language}"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open("/mount/resources/speech/corpora/CSS10/chinese/transcript.txt", encoding="utf8") as f:
            transcriptions = f.read()
        trans_lines = transcriptions.split("\n")
        for line in trans_lines:
            if line.strip() != "":
                path_to_transcript["/mount/resources/speech/corpora/CSS10/chinese/" + line.split("|")[0]] = line.split("|")[2]
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


# RUSSIAN

def build_path_to_transcript_dict_css10ru(re_cache=False):
    language = "russian"
    root = f"/mount/resources/speech/corpora/CSS10/{language}"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(f"/mount/resources/speech/corpora/CSS10/{language}/transcript.txt", encoding="utf8") as f:
            transcriptions = f.read()
        trans_lines = transcriptions.split("\n")
        for line in trans_lines:
            if line.strip() != "":
                path_to_transcript[f"/mount/resources/speech/corpora/CSS10/{language}/{line.split('|')[0]}"] = \
                    line.split("|")[2]
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


# HUNGARIAN

def build_path_to_transcript_dict_css10hu(re_cache=False):
    language = "hungarian"
    root = f"/mount/resources/speech/corpora/CSS10/{language}"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        language = "hungarian"
        with open(f"/mount/resources/speech/corpora/CSS10/{language}/transcript.txt", encoding="utf8") as f:
            transcriptions = f.read()
        trans_lines = transcriptions.split("\n")
        for line in trans_lines:
            if line.strip() != "":
                path_to_transcript[f"/mount/resources/speech/corpora/CSS10/{language}/{line.split('|')[0]}"] = \
                    line.split("|")[2]
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


# JAPANESE

def build_path_to_transcript_dict_captain_japanese(re_cache=False):
    root = "/mount/resources/speech/corpora/HiFiCaptainJapanese"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        with open(root + "/male/text/train_parallel.txt", encoding="utf8") as f:
            transcriptions = f.read()
        for line in transcriptions.split("\n"):
            if line.strip() != "":
                parsed_line = line.split()
                audio_path = parsed_line[0]
                transcript = parsed_line[1]
                audio_path = os.path.join(root, "male", "wav", "train_parallel", audio_path + ".wav")
                if os.path.exists(audio_path):
                    path_to_transcript[audio_path] = transcript.strip()
                else:
                    print(f"{audio_path} does not seem to exist!")
        with open(root + "/female/text/train_parallel.txt", encoding="utf8") as f:
            transcriptions = f.read()
        for line in transcriptions.split("\n"):
            if line.strip() != "":
                parsed_line = line.split()
                audio_path = parsed_line[0]
                transcript = parsed_line[1]
                audio_path = os.path.join(root, "female", "wav", "train_parallel", audio_path + ".wav")
                if os.path.exists(audio_path):
                    path_to_transcript[audio_path] = transcript.strip()
                else:
                    print(f"{audio_path} does not seem to exist!")
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_jvs(re_cache=False):
    root = "/mount/resources/speech/corpora/JVS/jvs_ver1"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = dict()
        for data_dir in os.listdir(root):
            if os.path.isdir(os.path.join(root, data_dir)):
                for data_type in ["parallel100", "nonpara30"]:
                    with open(os.path.join(root, data_dir, data_type, "transcripts_utf8.txt"), encoding="utf8") as f:
                        transcriptions = f.read()
                    for line in transcriptions.split("\n"):
                        if line.strip() != "":
                            parsed_line = line.split(":")
                            audio_path = parsed_line[0]
                            transcript = parsed_line[1]
                            audio_path = os.path.join(root, data_dir, data_type, "wav24kHz16bit", audio_path + ".wav")
                            if os.path.exists(audio_path):
                                path_to_transcript[audio_path] = transcript.strip()
                            else:
                                print(f"{audio_path} does not seem to exist!")
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


# OTHER


def build_path_to_transcript_dict_indicvoices_Assamese(re_cache=False):
    language = "Assamese"
    root = f"/mount/resources/speech/corpora/IndicVoicesR"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = indic_voices_template(root=root, lang=language)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_indicvoices_Bengali(re_cache=False):
    language = "Bengali"
    root = f"/mount/resources/speech/corpora/IndicVoicesR"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = indic_voices_template(root=root, lang=language)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_indicvoices_Bodo(re_cache=False):
    language = "Bodo"
    root = f"/mount/resources/speech/corpora/IndicVoicesR"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = indic_voices_template(root=root, lang=language)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_indicvoices_Dogri(re_cache=False):
    language = "Dogri"
    root = f"/mount/resources/speech/corpora/IndicVoicesR"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = indic_voices_template(root=root, lang=language)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_indicvoices_Gujarati(re_cache=False):
    language = "Gujarati"
    root = f"/mount/resources/speech/corpora/IndicVoicesR"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = indic_voices_template(root=root, lang=language)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_indicvoices_Hindi(re_cache=False):
    language = "Hindi"
    root = f"/mount/resources/speech/corpora/IndicVoicesR"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = indic_voices_template(root=root, lang=language)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_indicvoices_Kannada(re_cache=False):
    language = "Kannada"
    root = f"/mount/resources/speech/corpora/IndicVoicesR"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = indic_voices_template(root=root, lang=language)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_indicvoices_Kashmiri(re_cache=False):
    language = "Kashmiri"
    root = f"/mount/resources/speech/corpora/IndicVoicesR"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = indic_voices_template(root=root, lang=language)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_indicvoices_Konkani(re_cache=False):
    language = "Konkani"
    root = f"/mount/resources/speech/corpora/IndicVoicesR"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = indic_voices_template(root=root, lang=language)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_indicvoices_Maithili(re_cache=False):
    language = "Maithili"
    root = f"/mount/resources/speech/corpora/IndicVoicesR"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = indic_voices_template(root=root, lang=language)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_indicvoices_Malayalam(re_cache=False):
    language = "Malayalam"
    root = f"/mount/resources/speech/corpora/IndicVoicesR"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = indic_voices_template(root=root, lang=language)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_indicvoices_Manipuri(re_cache=False):
    language = "Manipuri"
    root = f"/mount/resources/speech/corpora/IndicVoicesR"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = indic_voices_template(root=root, lang=language)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_indicvoices_Marathi(re_cache=False):
    language = "Marathi"
    root = f"/mount/resources/speech/corpora/IndicVoicesR"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = indic_voices_template(root=root, lang=language)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_indicvoices_Nepali(re_cache=False):
    language = "Nepali"
    root = f"/mount/resources/speech/corpora/IndicVoicesR"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = indic_voices_template(root=root, lang=language)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_indicvoices_Odia(re_cache=False):
    language = "Odia"
    root = f"/mount/resources/speech/corpora/IndicVoicesR"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = indic_voices_template(root=root, lang=language)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_indicvoices_Punjabi(re_cache=False):
    language = "Punjabi"
    root = f"/mount/resources/speech/corpora/IndicVoicesR"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = indic_voices_template(root=root, lang=language)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_indicvoices_Sanskrit(re_cache=False):
    language = "Sanskrit"
    root = f"/mount/resources/speech/corpora/IndicVoicesR"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = indic_voices_template(root=root, lang=language)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_indicvoices_Santali(re_cache=False):
    language = "Santali"
    root = f"/mount/resources/speech/corpora/IndicVoicesR"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = indic_voices_template(root=root, lang=language)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_indicvoices_Sindhi(re_cache=False):
    language = "Sindhi"
    root = f"/mount/resources/speech/corpora/IndicVoicesR"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = indic_voices_template(root=root, lang=language)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_indicvoices_Tamil(re_cache=False):
    language = "Tamil"
    root = f"/mount/resources/speech/corpora/IndicVoicesR"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = indic_voices_template(root=root, lang=language)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_indicvoices_Telugu(re_cache=False):
    language = "Telugu"
    root = f"/mount/resources/speech/corpora/IndicVoicesR"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = indic_voices_template(root=root, lang=language)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_indicvoices_Urdu(re_cache=False):
    language = "Urdu"
    root = f"/mount/resources/speech/corpora/IndicVoicesR"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        path_to_transcript = indic_voices_template(root=root, lang=language)
        torch.save(path_to_transcript, cache_path)
    return torch.load(cache_path)


def build_file_list_singing_voice_audio_database(re_cache=False):
    root = "/mount/resources/speech/corpora/singing_voice_audio_dataset/monophonic"
    cache_path = os.path.join(root, "pttd_cache.pt")
    if not os.path.exists(cache_path) or re_cache:
        file_list = list()
        for corw in os.listdir(root):
            for singer in os.listdir(os.path.join(root, corw)):
                for audio in os.listdir(os.path.join(root, corw, singer)):
                    file_list.append(os.path.join(root, corw, singer, audio))
        torch.save(file_list, cache_path)
    return torch.load(cache_path)


def build_path_to_transcript_dict_nst_norwegian():
    root = '/resources/speech/corpora/NST_norwegian/pcm/cs'
    path_to_transcript = dict()
    audio_paths = sorted(list(Path(root).glob('*.pcm')))
    i = 0
    with open(Path(root, 'SCRIPTS/CTTS_core'), encoding='latin-1') as f:
        for line in f:
            transcript = line.strip().replace('\xad', '')
            path = str(audio_paths[i].absolute())
            path_to_transcript[path] = transcript
            i += 1
    return path_to_transcript


def build_path_to_transcript_dict_nst_swedish():
    root = '/resources/speech/corpora/NST_swedish/sw_pcms'
    path_to_transcript = dict()
    audio_paths = sorted(list(Path(root, 'mf').glob('*.pcm')))
    audio_paths.insert(4154, None)
    audio_paths.insert(5144, None)
    i = 0
    with open(Path(root, 'scripts/mf/sw_all'), encoding='latin-1') as f:
        for line in f:
            if i == 4154 or i == 5144:
                i += 1
                continue
            transcript = line.strip().replace('\xad', '')
            path = str(audio_paths[i].absolute())
            path_to_transcript[path] = transcript
            i += 1
    return path_to_transcript


def build_path_to_transcript_dict_nchlt_afr():
    root = '/resources/speech/corpora/nchlt_afr'
    return build_path_to_transcript_dict_nchlt_template(root, lang_code='afr')


def build_path_to_transcript_dict_nchlt_nbl():
    root = '/resources/speech/corpora/nchlt_nbl'
    return build_path_to_transcript_dict_nchlt_template(root, lang_code='nbl')


def build_path_to_transcript_dict_nchlt_nso():
    root = '/resources/speech/corpora/nchlt_nso'
    return build_path_to_transcript_dict_nchlt_template(root, lang_code='nso')


def build_path_to_transcript_dict_nchlt_sot():
    root = '/resources/speech/corpora/nchlt_sot'
    return build_path_to_transcript_dict_nchlt_template(root, lang_code='sot')


def build_path_to_transcript_dict_nchlt_ssw():
    root = '/resources/speech/corpora/nchlt_ssw'
    return build_path_to_transcript_dict_nchlt_template(root, lang_code='ssw')


def build_path_to_transcript_dict_nchlt_tsn():
    root = '/resources/speech/corpora/nchlt_tsn'
    return build_path_to_transcript_dict_nchlt_template(root, lang_code='tsn')


def build_path_to_transcript_dict_nchlt_tso():
    root = '/resources/speech/corpora/nchlt_tso'
    return build_path_to_transcript_dict_nchlt_template(root, lang_code='tso')


def build_path_to_transcript_dict_nchlt_ven():
    root = '/resources/speech/corpora/nchlt_ven'
    return build_path_to_transcript_dict_nchlt_template(root, lang_code='ven')


def build_path_to_transcript_dict_nchlt_xho():
    root = '/resources/speech/corpora/nchlt_xho'
    return build_path_to_transcript_dict_nchlt_template(root, lang_code='xho')


def build_path_to_transcript_dict_nchlt_zul():
    root = '/resources/speech/corpora/nchlt_zul'
    return build_path_to_transcript_dict_nchlt_template(root, lang_code='zul')


def build_path_to_transcript_dict_nchlt_template(root, lang_code):
    path_to_transcript = dict()
    base_dir = Path(root).parent

    for split in ['trn', 'tst']:
        tree = ET.parse(f'{root}/transcriptions/nchlt_{lang_code}.{split}.xml')
        tree_root = tree.getroot()
        for rec in tree_root.iter('recording'):
            transcript = rec.find('orth').text
            if '[s]' in transcript:
                continue
            path = str(base_dir / rec.get('audio'))
            path_to_transcript[path] = transcript

    return path_to_transcript


def build_path_to_transcript_dict_bibletts_akuapem_twi():
    path_to_transcript = dict()
    root = '/resources/speech/corpora/BibleTTS/akuapem-twi'
    for split in ['train', 'dev', 'test']:
        for book in Path(root, split).glob('*'):
            for textfile in book.glob('*.txt'):
                with open(textfile, 'r', encoding='utf-8') as f:
                    text = ' '.join([line.strip() for line in f])  # should usually be only one line anyway
                path_to_transcript[textfile.with_suffix('.flac')] = text

    return path_to_transcript


def build_path_to_transcript_dict_bembaspeech():
    root = '/resources/speech/corpora/BembaSpeech/bem'
    path_to_transcript = dict()

    for split in ['train', 'dev', 'test']:
        with open(Path(root, f'{split}.tsv'), 'r', encoding='utf-8') as f:
            reader = DictReader(f, delimiter='\t')
            for row in reader:
                path_to_transcript[str(Path(root, 'audio', row['audio']))] = row['sentence']

    return path_to_transcript


def build_path_to_transcript_dict_alffa_sw():
    root = '/resources/speech/corpora/ALFFA/data_broadcastnews_sw/data'

    path_to_transcript = build_path_to_transcript_dict_kaldi_template(root=root, split='train', replace_in_path=('asr_swahili/data/', ''))
    path_to_transcript.update(build_path_to_transcript_dict_kaldi_template(root=root, split='test', replace_in_path=('/my_dir/wav', 'test/wav5')))
    return path_to_transcript


def build_path_to_transcript_dict_alffa_am():
    root = '/resources/speech/corpora/ALFFA/data_readspeech_am/data'

    path_to_transcript = build_path_to_transcript_dict_kaldi_template(root=root, split='train', replace_in_path=('/home/melese/kaldi/data/', ''))
    path_to_transcript.update(build_path_to_transcript_dict_kaldi_template(root=root, split='test', replace_in_path=('/home/melese/kaldi/data/', '')))

    return path_to_transcript


def build_path_to_transcript_dict_alffa_wo():
    root = '/resources/speech/corpora/ALFFA/data_readspeech_wo/data'
    path_to_transcript = dict()

    for split in ['train', 'dev', 'test']:
        with open(Path(root, split, 'text'), 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split()
                file = line[0]
                text = ' '.join(line[1:])
                number = file.split('_')[1]
                path_to_transcript[str(Path(root, split, number, f'{file}.wav'))] = text

    return path_to_transcript


def build_path_to_transcript_dict_malayalam():
    root = '/resources/speech/corpora/malayalam'
    path_to_transcript = dict()

    for gender in ['female', 'male']:
        with open(Path(root, f'line_index_{gender}.tsv'), 'r', encoding='utf-8') as f:
            for line in f:
                file, text = line.strip().split('\t')
                path_to_transcript[str(Path(root, gender, f'{file}.wav'))] = text

    return path_to_transcript


def build_path_to_transcript_dict_msc():
    root = '/resources/speech/corpora/msc_reviewed_speech'
    path_to_transcript = dict()

    with open(Path(root, f'metadata.tsv'), 'r', encoding='utf-8') as f:
        reader = DictReader(f, delimiter='\t')
        for row in reader:
            path_to_transcript[str(Path(root, row['speechpath']))] = row['transcript']

    return path_to_transcript


def build_path_to_transcript_dict_chuvash():
    root = '/resources/speech/corpora/chuvash'
    path_to_transcript = dict()

    for textfile in Path(root, 'transcripts', 'txt').glob('*.txt'):
        with open(textfile, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split()
                text = ' '.join(line[1:]).replace('«', '').replace('»', '')
                path = Path(root, 'audio', 'split', f'trim_clean_{textfile.stem}.{line[0]}.flac')
                if path.exists():
                    path_to_transcript[str(path)] = text

    return path_to_transcript


def build_path_to_transcript_dict_iban():
    root = '/resources/speech/corpora/iban/data'
    path_to_transcript = build_path_to_transcript_dict_kaldi_template(root, 'train', replace_in_path=(
        'asr_iban/data/', ''))
    path_to_transcript.update(build_path_to_transcript_dict_kaldi_template(root, 'dev', replace_in_path=(
        'asr_iban/data/', '')))
    return path_to_transcript


def build_path_to_transcript_dict_kaldi_template(root, split, replace_in_path=None):
    path_to_transcript = dict()

    wav_scp = {}
    with open(Path(root, split, 'wav.scp'), 'r') as f:
        for line in f:
            wav_id, wav_path = line.split()
            if replace_in_path:
                wav_path = wav_path.replace(replace_in_path[0], replace_in_path[1])
            wav_scp[wav_id] = str(Path(root, wav_path))

    with open(Path(root, split, 'text'), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split()
            wav_id = line[0]
            text = ' '.join(line[1:])
            if '<' in text:  # ignore all <UNK> utterance etc.
                continue
            path_to_transcript[wav_scp[wav_id]] = text

    return path_to_transcript


def build_path_to_transcript_dict_sundanese_speech():
    root = '/resources/speech/corpora/sundanese_speech/asr_sundanese'
    return build_path_to_transcript_dict_south_asian_languages_template(root)


def build_path_to_transcript_dict_sinhala_speech():
    root = '/resources/speech/corpora/sinhala_speech/asr_sinhala'
    return build_path_to_transcript_dict_south_asian_languages_template(root)


def build_path_to_transcript_dict_bengali_speech():
    root = '/resources/speech/corpora/bengali_speech/asr_bengali'
    return build_path_to_transcript_dict_south_asian_languages_template(root)


def build_path_to_transcript_dict_nepali_speech():
    root = '/resources/speech/corpora/nepali_speech/asr_nepali'
    return build_path_to_transcript_dict_south_asian_languages_template(root)


def build_path_to_transcript_dict_javanese_speech():
    root = '/resources/speech/corpora/javanese_speech/asr_javanese'
    return build_path_to_transcript_dict_south_asian_languages_template(root)


def build_path_to_transcript_dict_south_asian_languages_template(root):
    path_to_transcript = dict()

    with open(Path(root, 'utt_spk_text.tsv'), 'r', encoding='utf-8') as f:
        for line in f:
            utt, spk, text = line.strip().split('\t')
            dir_tag = utt[:2]
            path_to_transcript[str(Path(root, 'data', dir_tag, f'{utt}.flac'))] = text

    return path_to_transcript


def build_path_to_transcript_dict_african_voices_kenyan_afv():
    root = '/resources/speech/corpora/AfricanVoices/afv_enke'
    return build_path_to_transcript_dict_african_voices_template(root)


def build_path_to_transcript_dict_african_voices_fon_alf():
    root = '/resources/speech/corpora/AfricanVoices/fon_alf'
    return build_path_to_transcript_dict_african_voices_template(root)


def build_path_to_transcript_dict_african_voices_hausa_cmv():
    main_root = '/resources/speech/corpora/AfricanVoices'
    path_to_transcript = build_path_to_transcript_dict_african_voices_template(f'{main_root}/hau_cmv_f')
    path_to_transcript.update(build_path_to_transcript_dict_african_voices_template(f'{main_root}/hau_cmv_m'))
    return path_to_transcript


def build_path_to_transcript_dict_african_voices_ibibio_lst():
    root = '/resources/speech/corpora/AfricanVoices/ibb_lst'
    return build_path_to_transcript_dict_african_voices_template(root)


def build_path_to_transcript_dict_african_voices_kikuyu_opb():
    root = '/resources/speech/corpora/AfricanVoices/kik_opb'
    return build_path_to_transcript_dict_african_voices_template(root)


def build_path_to_transcript_dict_african_voices_lingala_opb():
    root = '/resources/speech/corpora/AfricanVoices/lin_opb'
    return build_path_to_transcript_dict_african_voices_template(root)


def build_path_to_transcript_dict_african_voices_ganda_cmv():
    root = '/resources/speech/corpora/AfricanVoices/lug_cmv'
    return build_path_to_transcript_dict_african_voices_template(root)


def build_path_to_transcript_dict_african_voices_luo_afv():
    root = '/resources/speech/corpora/AfricanVoices/luo_afv'
    return build_path_to_transcript_dict_african_voices_template(root)


def build_path_to_transcript_dict_african_voices_luo_opb():
    root = '/resources/speech/corpora/AfricanVoices/luo_opb'
    return build_path_to_transcript_dict_african_voices_template(root)


def build_path_to_transcript_dict_african_voices_swahili_llsti():
    root = '/resources/speech/corpora/AfricanVoices/swa_llsti'
    return build_path_to_transcript_dict_african_voices_template(root)


def build_path_to_transcript_dict_african_voices_suba_afv():
    root = '/resources/speech/corpora/AfricanVoices/sxb_afv'
    return build_path_to_transcript_dict_african_voices_template(root)


def build_path_to_transcript_dict_african_voices_wolof_alf():
    root = '/resources/speech/corpora/AfricanVoices/wol_alf'
    return build_path_to_transcript_dict_african_voices_template(root)


def build_path_to_transcript_dict_african_voices_yoruba_opb():
    root = '/resources/speech/corpora/AfricanVoices/yor_opb'
    return build_path_to_transcript_dict_african_voices_template(root)


def build_path_to_transcript_dict_african_voices_template(root):
    path_to_transcript = dict()

    with open(Path(root, 'txt.done.data'), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.replace('\\"', "'").split('"')
            text = line[1]
            file = line[0].split()[-1]
            path_to_transcript[str(Path(root, 'wav', f'{file}.wav'))] = text

    return path_to_transcript


def build_path_to_transcript_dict_zambezi_voice_nyanja():
    root = '/resources/speech/corpora/ZambeziVoice/nyanja/nya'
    return build_path_to_transcript_dict_zambezi_voice_template(root)


def build_path_to_transcript_dict_zambezi_voice_lozi():
    root = '/resources/speech/corpora/ZambeziVoice/lozi/loz'
    return build_path_to_transcript_dict_zambezi_voice_template(root)


def build_path_to_transcript_dict_zambezi_voice_tonga():
    root = '/resources/speech/corpora/ZambeziVoice/tonga/toi'
    return build_path_to_transcript_dict_zambezi_voice_template(root)


def build_path_to_transcript_dict_zambezi_voice_template(root):
    path_to_transcript = dict()

    for split in ['train', 'dev', 'test']:
        with open(Path(root, f'{split}.tsv'), 'r', encoding='utf-8') as f:
            reader = DictReader(f, delimiter='\t')
            for row in reader:
                path_to_transcript[str(Path(root, 'audio', row['audio_id']))] = row['sentence'].strip()

    return path_to_transcript


def build_path_to_transcript_dict_fleurs_template(root):
    path_to_transcript = dict()

    for split in ['train', 'dev', 'test']:
        with open(Path(root, f'{split}.tsv'), 'r', encoding='utf-8') as f:
            reader = DictReader(f, delimiter='\t', fieldnames=['id', 'filename', 'transcription_raw',
                                                               'transcription', 'words', 'speaker', 'gender'])
            for row in reader:
                path_to_transcript[str(Path(root, 'audio', split, row['filename']))] = row['transcription_raw'].strip()

    return path_to_transcript


def build_path_to_transcript_dict_fleurs_afrikaans():
    root = '/resources/speech/corpora/fleurs/af_za'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_amharic():
    root = '/resources/speech/corpora/fleurs/am_et'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_arabic():
    root = '/resources/speech/corpora/fleurs/ar_eg'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_assamese():
    root = '/resources/speech/corpora/fleurs/as_in'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_asturian():
    root = '/resources/speech/corpora/fleurs/ast_es'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_azerbaijani():
    root = '/resources/speech/corpora/fleurs/az_az'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_belarusian():
    root = '/resources/speech/corpora/fleurs/be_by'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_bulgarian():
    root = '/resources/speech/corpora/fleurs/bg_bg'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_bengali():
    root = '/resources/speech/corpora/fleurs/bn_in'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_bosnian():
    root = '/resources/speech/corpora/fleurs/bs_ba'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_catalan():
    root = '/resources/speech/corpora/fleurs/ca_es'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_cebuano():
    root = '/resources/speech/corpora/fleurs/ceb_ph'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_sorani_kurdish():
    root = '/resources/speech/corpora/fleurs/ckb_iq'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_mandarin():
    root = '/resources/speech/corpora/fleurs/cmn_hans_cn'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_czech():
    root = '/resources/speech/corpora/fleurs/cs_cz'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_welsh():
    root = '/resources/speech/corpora/fleurs/cy_gb'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_danish():
    root = '/resources/speech/corpora/fleurs/da_dk'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_german():
    root = '/resources/speech/corpora/fleurs/de_de'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_greek():
    root = '/resources/speech/corpora/fleurs/el_gr'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_english():
    root = '/resources/speech/corpora/fleurs/en_us'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_spanish():
    root = '/resources/speech/corpora/fleurs/es_419'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_estonian():
    root = '/resources/speech/corpora/fleurs/et_ee'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_persian():
    root = '/resources/speech/corpora/fleurs/fa_ir'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_fula():
    root = '/resources/speech/corpora/fleurs/ff_sn'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_finnish():
    root = '/resources/speech/corpora/fleurs/fi_fi'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_filipino():
    root = '/resources/speech/corpora/fleurs/fil_ph'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_french():
    root = '/resources/speech/corpora/fleurs/fr_fr'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_irish():
    root = '/resources/speech/corpora/fleurs/ga_ie'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_galician():
    root = '/resources/speech/corpora/fleurs/gl_es'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_gujarati():
    root = '/resources/speech/corpora/fleurs/gu_in'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_hausa():
    root = '/resources/speech/corpora/fleurs/ha_ng'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_hebrew():
    root = '/resources/speech/corpora/fleurs/he_il'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_hindi():
    root = '/resources/speech/corpora/fleurs/hi_in'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_croatian():
    root = '/resources/speech/corpora/fleurs/hr_hr'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_hungarian():
    root = '/resources/speech/corpora/fleurs/hu_hu'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_armenian():
    root = '/resources/speech/corpora/fleurs/hy_am'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_indonesian():
    root = '/resources/speech/corpora/fleurs/id_id'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_igbo():
    root = '/resources/speech/corpora/fleurs/ig_ng'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_icelandic():
    root = '/resources/speech/corpora/fleurs/is_is'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_italian():
    root = '/resources/speech/corpora/fleurs/it_it'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_japanese():
    root = '/resources/speech/corpora/fleurs/ja_jp'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_javanese():
    root = '/resources/speech/corpora/fleurs/jv_id'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_georgian():
    root = '/resources/speech/corpora/fleurs/ka_ge'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_kamba():
    root = '/resources/speech/corpora/fleurs/kam_ke'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_kabuverdianu():
    root = '/resources/speech/corpora/fleurs/kea_cv'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_kazakh():
    root = '/resources/speech/corpora/fleurs/kk_kz'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_khmer():
    root = '/resources/speech/corpora/fleurs/km_kh'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_kannada():
    root = '/resources/speech/corpora/fleurs/kn_in'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_korean():
    root = '/resources/speech/corpora/fleurs/ko_kr'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_kyrgyz():
    root = '/resources/speech/corpora/fleurs/ky_kg'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_luxembourgish():
    root = '/resources/speech/corpora/fleurs/lb_lu'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_ganda():
    root = '/resources/speech/corpora/fleurs/lg_ug'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_lingala():
    root = '/resources/speech/corpora/fleurs/ln_cd'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_lao():
    root = '/resources/speech/corpora/fleurs/lo_la'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_lithuanian():
    root = '/resources/speech/corpora/fleurs/lt_lt'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_luo():
    root = '/resources/speech/corpora/fleurs/luo_ke'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_latvian():
    root = '/resources/speech/corpora/fleurs/lv_lv'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_maori():
    root = '/resources/speech/corpora/fleurs/mi_nz'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_macedonian():
    root = '/resources/speech/corpora/fleurs/mk_mk'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_malayalam():
    root = '/resources/speech/corpora/fleurs/ml_in'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_mongolian():
    root = '/resources/speech/corpora/fleurs/mn_mn'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_marathi():
    root = '/resources/speech/corpora/fleurs/mr_in'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_malay():
    root = '/resources/speech/corpora/fleurs/ms_my'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_maltese():
    root = '/resources/speech/corpora/fleurs/mt_mt'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_burmese():
    root = '/resources/speech/corpora/fleurs/my_mm'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_norwegian():
    root = '/resources/speech/corpora/fleurs/nb_no'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_nepali():
    root = '/resources/speech/corpora/fleurs/ne_np'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_dutch():
    root = '/resources/speech/corpora/fleurs/nl_nl'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_northern_sotho():
    root = '/resources/speech/corpora/fleurs/nso_za'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_nyanja():
    root = '/resources/speech/corpora/fleurs/ny_mw'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_occitan():
    root = '/resources/speech/corpora/fleurs/oc_fr'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_oroma():
    root = '/resources/speech/corpora/fleurs/om_et'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_oriya():
    root = '/resources/speech/corpora/fleurs/or_in'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_punjabi():
    root = '/resources/speech/corpora/fleurs/pa_in'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_polish():
    root = '/resources/speech/corpora/fleurs/pl_pl'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_pashto():
    root = '/resources/speech/corpora/fleurs/ps_af'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_portuguese():
    root = '/resources/speech/corpora/fleurs/pt_br'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_romanian():
    root = '/resources/speech/corpora/fleurs/ro_ro'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_russian():
    root = '/resources/speech/corpora/fleurs/ru_ru'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_sindhi():
    root = '/resources/speech/corpora/fleurs/sd_in'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_slovak():
    root = '/resources/speech/corpora/fleurs/sk_sk'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_slovenian():
    root = '/resources/speech/corpora/fleurs/sl_si'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_shona():
    root = '/resources/speech/corpora/fleurs/sn_zw'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_somali():
    root = '/resources/speech/corpora/fleurs/so_so'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_serbian():
    root = '/resources/speech/corpora/fleurs/sr_rs'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_swedish():
    root = '/resources/speech/corpora/fleurs/sv_se'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_swahili():
    root = '/resources/speech/corpora/fleurs/sw_ke'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_tamil():
    root = '/resources/speech/corpora/fleurs/ta_in'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_telugu():
    root = '/resources/speech/corpora/fleurs/te_in'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_tajik():
    root = '/resources/speech/corpora/fleurs/tg_tj'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_thai():
    root = '/resources/speech/corpora/fleurs/th_th'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_turkish():
    root = '/resources/speech/corpora/fleurs/tr_tr'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_ukrainian():
    root = '/resources/speech/corpora/fleurs/uk_ua'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_umbundu():
    root = '/resources/speech/corpora/fleurs/umb_ao'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_urdu():
    root = '/resources/speech/corpora/fleurs/ur_pk'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_uzbek():
    root = '/resources/speech/corpora/fleurs/uz_uz'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_vietnamese():
    root = '/resources/speech/corpora/fleurs/vi_vn'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_wolof():
    root = '/resources/speech/corpora/fleurs/wo_sn'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_xhosa():
    root = '/resources/speech/corpora/fleurs/xh_za'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_yoruba():
    root = '/resources/speech/corpora/fleurs/yo_ng'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_cantonese():
    root = '/resources/speech/corpora/fleurs/yue_hant_hk'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_fleurs_zulu():
    root = '/resources/speech/corpora/fleurs/zu_za'
    return build_path_to_transcript_dict_fleurs_template(root)


def build_path_to_transcript_dict_living_audio_dataset_template(root):
    path_to_transcript = dict()
    tree = ET.parse(f'{root}/text.xml')
    tree_root = tree.getroot()

    for rec in tree_root.iter('recording_script'):
        for file in rec.iter('fileid'):
            path_to_transcript[str(Path(root, '48000_orig', f'{file.get("id")}.wav'))] = file.text.strip()

    return path_to_transcript


def build_path_to_transcript_dict_living_audio_dataset_irish():
    root = '/resources/speech/corpora/LivingAudioDataset/ga'
    return build_path_to_transcript_dict_living_audio_dataset_template(root)


def build_path_to_transcript_dict_living_audio_dataset_dutch():
    root = '/resources/speech/corpora/LivingAudioDataset/nl'
    return build_path_to_transcript_dict_living_audio_dataset_template(root)


def build_path_to_transcript_dict_living_audio_dataset_russian():
    root = '/resources/speech/corpora/LivingAudioDataset/ru'
    return build_path_to_transcript_dict_living_audio_dataset_template(root)


def build_path_to_transcript_dict_romanian_db():
    root = '/resources/speech/corpora/RomanianDB'
    path_to_transcript = dict()

    for split in ['training', 'testing', 'elena', 'georgiana']:
        for transcript in Path(root, split, 'text').glob('*.txt'):
            subset = transcript.stem
            with open(transcript, 'r', encoding='utf-8') as f:
                for line in f:
                    fileid = line.strip()[:2]
                    if len(fileid) == 2:
                        fileid = '0' + fileid
                    text = line.strip()[5:]
                    if split == 'elena':
                        path = f'ele_{subset}_{fileid}.wav'
                    elif split == 'georgiana':
                        path = f'geo_{subset}_{fileid}.wav'
                    else:
                        path = f'adr_{subset}_{fileid}.wav'
                    path_to_transcript[str(Path(root, split, 'wav', subset, path))] = text

    return path_to_transcript


def build_path_to_transcript_dict_shemo():
    root = '/resources/speech/corpora/ShEMO'
    path_to_transcript = dict()

    with open('/resources/speech/corpora/ShEMO/shemo.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    for fileid, file_info in data.items():
        path = Path(root, file_info['gender'], f'{fileid}.wav')
        if path.exists():
            path_to_transcript[str(path)] = file_info['transcript']

    return path_to_transcript


def build_path_to_transcript_dict_mslt_template(root, lang='en'):
    path_to_transcript = dict()

    for split in Path(root).glob('*'):
        if split.is_dir():
            for audio_file in split.glob('*.wav'):
                text_file = str(audio_file).replace(f'T0.{lang}.wav', f'T1.{lang}.snt')
                with open(text_file, 'r', encoding='utf-16') as f:
                    for line in f:
                        text = line.strip()  # should have only one line
                        if '<' in text or '[' in text:
                            # ignore all utterances with special parts like [laughter] or <UNIN/>
                            continue
                        path_to_transcript[str(audio_file)] = text
                        break

    return path_to_transcript


def build_path_to_transcript_dict_mslt_english():
    root = '/resources/speech/corpora/MSLT/Data/EN'
    return build_path_to_transcript_dict_mslt_template(root, lang='en')


def build_path_to_transcript_dict_mslt_japanese():
    root = '/resources/speech/corpora/MSLT/Data/JA'
    return build_path_to_transcript_dict_mslt_template(root, lang='jp')


def build_path_to_transcript_dict_mslt_chinese():
    root = '/resources/speech/corpora/MSLT/Data/ZH'
    return build_path_to_transcript_dict_mslt_template(root, lang='ch')


def build_path_to_transcript_dict_rajasthani_hindi_speech():
    root = '/resources/speech/corpora/Rajasthani_Hindi_Speech/Hindi-Speech-Data'
    path_to_transcript = dict()

    for audio_file in Path(root).glob('*.3gp'):
        with open(audio_file.with_suffix('.txt'), 'r', encoding='utf-8') as f:
            for line in f:  # should only be one line
                text = line.strip()
        path_to_transcript[str(audio_file)] = text

    return path_to_transcript


def build_path_to_transcript_dict_cmu_arctic():
    root = '/resources/speech/corpora/cmu_arctic'
    path_to_transcript = dict()

    for speaker_dir in Path(root).glob('*'):
        if speaker_dir.is_dir():
            with open(Path(speaker_dir, 'etc', 'txt.done.data'), 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.replace('\\"', "'").split('"')
                    text = line[1]
                    file = line[0].split()[-1]
                    path_to_transcript[str(Path(speaker_dir, 'wav', f'{file}.wav'))] = text

    return path_to_transcript


def build_path_to_transcript_dict_sevil_tatar():
    root = '/resources/speech/corpora/sevil_tatar/sevil'
    path_to_transcript = dict()

    with open(Path(root, 'metadata.jsonl'), 'r', encoding='utf-8') as f:
        for line in f:
            meta = json.loads(line)
            path_to_transcript[str(Path(root, meta['file']))] = meta['orig_text'].strip().replace('\xad', '')

    return path_to_transcript


def build_path_to_transcript_dict_clartts():
    root = '/resources/speech/corpora/ClArTTS'
    path_to_transcript = dict()

    with open(Path(root, 'training.txt'), 'r', encoding='utf-16') as f:
        for line in f:
            fileid, transcript = line.strip().split('|')
            path_to_transcript[str(Path(root, 'wav', 'train', f'{fileid}.wav'))] = transcript

    with open(Path(root, 'validation.txt'), 'r', encoding='utf-16') as f:
        for line in f:
            fileid, transcript = line.strip().split('|')
            path_to_transcript[str(Path(root, 'wav', 'val', f'{fileid}.wav'))] = transcript

    return path_to_transcript


def build_path_to_transcript_dict_snow_mountain_template(root, lang):
    path_to_transcript = dict()

    for split in ['train_full', 'val_full', 'test_common']:
        with open(Path(root, 'experiments', lang, f'{split}.csv'), 'r', encoding='utf-8') as f:
            reader = DictReader(f, delimiter=',')
            for row in reader:
                path = row['path'].replace('data/', f'{root}/')
                path_to_transcript[path] = row['sentence'].strip()

    return path_to_transcript


def build_path_to_transcript_dict_snow_mountain_bhadrawahi():
    root = '/resources/speech/corpora/snow_mountain'
    language = 'bhadrawahi'
    return build_path_to_transcript_dict_snow_mountain_template(root, language)


def build_path_to_transcript_dict_snow_mountain_bilaspuri():
    root = '/resources/speech/corpora/snow_mountain'
    language = 'bilaspuri'
    return build_path_to_transcript_dict_snow_mountain_template(root, language)


def build_path_to_transcript_dict_snow_mountain_dogri():
    root = '/resources/speech/corpora/snow_mountain'
    language = 'dogri'
    return build_path_to_transcript_dict_snow_mountain_template(root, language)


def build_path_to_transcript_dict_snow_mountain_gaddi():
    root = '/resources/speech/corpora/snow_mountain'
    language = 'gaddi'
    return build_path_to_transcript_dict_snow_mountain_template(root, language)


def build_path_to_transcript_dict_snow_mountain_haryanvi():
    root = '/resources/speech/corpora/snow_mountain'
    language = 'haryanvi'
    return build_path_to_transcript_dict_snow_mountain_template(root, language)


def build_path_to_transcript_dict_snow_mountain_hindi():
    root = '/resources/speech/corpora/snow_mountain'
    language = 'hindi'
    return build_path_to_transcript_dict_snow_mountain_template(root, language)


def build_path_to_transcript_dict_snow_mountain_kangri():
    root = '/resources/speech/corpora/snow_mountain'
    language = 'kangri'
    return build_path_to_transcript_dict_snow_mountain_template(root, language)


def build_path_to_transcript_dict_snow_mountain_kannada():
    root = '/resources/speech/corpora/snow_mountain'
    language = 'kannada'
    return build_path_to_transcript_dict_snow_mountain_template(root, language)


def build_path_to_transcript_dict_snow_mountain_kulvi():
    root = '/resources/speech/corpora/snow_mountain'
    language = 'kulvi'
    return build_path_to_transcript_dict_snow_mountain_template(root, language)


def build_path_to_transcript_dict_snow_mountain_kulvi_outer_seraji():
    root = '/resources/speech/corpora/snow_mountain'
    language = 'kulvi_outer_seraji'
    return build_path_to_transcript_dict_snow_mountain_template(root, language)


def build_path_to_transcript_dict_snow_mountain_malayalam():
    root = '/resources/speech/corpora/snow_mountain'
    language = 'malayalam'
    return build_path_to_transcript_dict_snow_mountain_template(root, language)


def build_path_to_transcript_dict_snow_mountain_mandeali():
    root = '/resources/speech/corpora/snow_mountain'
    language = 'mandeali'
    return build_path_to_transcript_dict_snow_mountain_template(root, language)


def build_path_to_transcript_dict_snow_mountain_pahari_mahasui():
    root = '/resources/speech/corpora/snow_mountain'
    language = 'pahari_mahasui'
    return build_path_to_transcript_dict_snow_mountain_template(root, language)


def build_path_to_transcript_dict_snow_mountain_tamil():
    root = '/resources/speech/corpora/snow_mountain'
    language = 'tamil'
    return build_path_to_transcript_dict_snow_mountain_template(root, language)


def build_path_to_transcript_dict_snow_mountain_telugu():
    root = '/resources/speech/corpora/snow_mountain'
    language = 'telugu'
    return build_path_to_transcript_dict_snow_mountain_template(root, language)


def build_path_to_transcript_dict_ukrainian_lada():
    root = '/resources/speech/corpora/ukrainian_lada/dataset_lada/accept'
    path_to_transcript = dict()

    with open(Path(root, 'metadata.jsonl'), 'r', encoding='utf-8') as f:
        for line in f:
            meta = json.loads(line)
            path_to_transcript[str(Path(root, meta['file']).with_suffix('.wav'))] = meta['orig_text'].strip().replace('\xad', '')

    return path_to_transcript


def build_path_to_transcript_dict_m_ailabs_template(root):
    path_to_transcript = dict()

    for gender_dir in Path(root).glob('*'):
        if not gender_dir.is_dir():
            continue
        for speaker_dir in gender_dir.glob('*'):
            if not speaker_dir.is_dir():
                continue
            if (speaker_dir / 'wavs').exists():
                with open(Path(speaker_dir, 'metadata.csv'), 'r', encoding='utf-8') as f:
                    for line in f:
                        fileid, text, text_norm = line.strip().split('|')
                        path = Path(speaker_dir, 'wavs', f'{fileid}.wav')
                        if path.exists():
                            path_to_transcript[str(path)] = text_norm
            else:

                for session_dir in speaker_dir.glob('*'):
                    if not session_dir.is_dir():
                        continue
                    with open(Path(session_dir, 'metadata.csv'), 'r', encoding='utf-8') as f:
                        for line in f:
                            fileid, text, text_norm = line.strip().split('|')
                            path = Path(session_dir, 'wavs', f'{fileid}.wav')
                            if path.exists():
                                path_to_transcript[str(path)] = text_norm

    return path_to_transcript


def build_path_to_transcript_dict_m_ailabs_german():
    root = '/resources/speech/corpora/m-ailabs-speech/de_DE'
    return build_path_to_transcript_dict_m_ailabs_template(root)


def build_path_to_transcript_dict_m_ailabs_uk_english():
    root = '/resources/speech/corpora/m-ailabs-speech/en_UK'
    return build_path_to_transcript_dict_m_ailabs_template(root)


def build_path_to_transcript_dict_m_ailabs_us_english():
    root = '/resources/speech/corpora/m-ailabs-speech/en_US'
    return build_path_to_transcript_dict_m_ailabs_template(root)


def build_path_to_transcript_dict_m_ailabs_spanish():
    root = '/resources/speech/corpora/m-ailabs-speech/es_ES'
    return build_path_to_transcript_dict_m_ailabs_template(root)


def build_path_to_transcript_dict_m_ailabs_french():
    root = '/resources/speech/corpora/m-ailabs-speech/fr_FR'
    return build_path_to_transcript_dict_m_ailabs_template(root)


def build_path_to_transcript_dict_m_ailabs_italian():
    root = '/resources/speech/corpora/m-ailabs-speech/it_IT'
    return build_path_to_transcript_dict_m_ailabs_template(root)


def build_path_to_transcript_dict_m_ailabs_polish():
    root = '/resources/speech/corpora/m-ailabs-speech/pl_PL'
    return build_path_to_transcript_dict_m_ailabs_template(root)


def build_path_to_transcript_dict_m_ailabs_russian():
    root = '/resources/speech/corpora/m-ailabs-speech/ru_RU'
    return build_path_to_transcript_dict_m_ailabs_template(root)


def build_path_to_transcript_dict_m_ailabs_ukrainian():
    root = '/resources/speech/corpora/m-ailabs-speech/uk_UK'
    return build_path_to_transcript_dict_m_ailabs_template(root)


def build_path_to_transcript_dict_cml_tts_template(root):
    path_to_transcript = dict()

    for split in ['train', 'dev', 'test']:
        with open(Path(root, f'{split}.csv'), 'r', encoding='utf-8') as f:
            reader = DictReader(f, delimiter='|')
            for row in reader:
                path_to_transcript[str(Path(root, row['wav_filename']))] = row['transcript'].strip()

    return path_to_transcript


def build_path_to_transcript_dict_cml_tts_dutch():
    root = '/resources/speech/corpora/cml_tts/cml_tts_dataset_dutch_v0.1'
    return build_path_to_transcript_dict_cml_tts_template(root)


def build_path_to_transcript_dict_cml_tts_french():
    root = '/resources/speech/corpora/cml_tts/cml_tts_dataset_french_v0.1'
    return build_path_to_transcript_dict_cml_tts_template(root)


def build_path_to_transcript_dict_cml_tts_german():
    root = '/resources/speech/corpora/cml_tts/cml_tts_dataset_german_v0.1'
    return build_path_to_transcript_dict_cml_tts_template(root)


def build_path_to_transcript_dict_cml_tts_italian():
    root = '/resources/speech/corpora/cml_tts/cml_tts_dataset_italian_v0.1'
    return build_path_to_transcript_dict_cml_tts_template(root)


def build_path_to_transcript_dict_cml_tts_polish():
    root = '/resources/speech/corpora/cml_tts/cml_tts_dataset_polish_v0.1'
    return build_path_to_transcript_dict_cml_tts_template(root)


def build_path_to_transcript_dict_cml_tts_portuguese():
    root = '/resources/speech/corpora/cml_tts/cml_tts_dataset_portuguese_v0.1'
    return build_path_to_transcript_dict_cml_tts_template(root)


def build_path_to_transcript_dict_cml_tts_spanish():
    root = '/resources/speech/corpora/cml_tts/cml_tts_dataset_spanish_v0.1'
    return build_path_to_transcript_dict_cml_tts_template(root)


def build_path_to_transcript_dict_mms_template(lang, root='/resources/speech/corpora/mms_synthesized_bible_speech'):
    path_to_transcript = dict()

    i = 0
    with open(Path(root, 'bible_texts', f'{lang}.txt'), 'r', encoding='utf-8') as f:
        for line in f:
            path = Path(root, 'bible_audios', lang, f'{i}.wav')
            if path.exists():
                path_to_transcript[str(path)] = line.strip()
                i += 1

    return path_to_transcript


if __name__ == '__main__':
    pass
