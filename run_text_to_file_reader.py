import os

import torch

from InferenceInterfaces.PortaSpeechInterface import PortaSpeechInterface


def read_texts(model_id, sentence, filename, device="cpu", language="en", speaker_reference=None):
    tts = PortaSpeechInterface(device=device, tts_model_path=model_id)
    tts.set_language(language)
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if type(sentence) == str:
        sentence = [sentence]
    tts.read_to_file(text_list=sentence, file_location=filename)
    del tts


def read_texts_as_ensemble(model_id, sentence, filename, device="cpu", language="en", amount=10):
    """
    for this function, the filename should NOT contain the .wav ending, it's added automatically
    """
    tts = PortaSpeechInterface(device=device, tts_model_path=model_id)
    tts.set_language(language)
    if type(sentence) == str:
        sentence = [sentence]
    for index in range(amount):
        tts.default_utterance_embedding = torch.zeros(256).float().random_(-40, 40).to(device)
        tts.read_to_file(text_list=sentence, file_location=filename + f"_{index}" + ".wav")


def read_harvard_sentences(model_id, device):
    tts = PortaSpeechInterface(device=device, tts_model_path=model_id)

    with open("Utility/test_sentences_combined_3.txt", "r", encoding="utf8") as f:
        sents = f.read().split("\n")
    output_dir = "audios/harvard_03_{}".format(model_id)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for index, sent in enumerate(sents):
        tts.read_to_file(text_list=[sent], file_location=output_dir + "/{}.wav".format(index))

    with open("Utility/test_sentences_combined_6.txt", "r", encoding="utf8") as f:
        sents = f.read().split("\n")
    output_dir = "audios/harvard_06_{}".format(model_id)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for index, sent in enumerate(sents):
        tts.read_to_file(text_list=[sent], file_location=output_dir + "/{}.wav".format(index))


def le_corbeau_et_le_renard(version, model_id="Meta", exec_device="cpu"):
    os.makedirs("audios", exist_ok=True)

    read_texts(model_id=model_id,
               sentence=["Maître Corbeau, sur un arbre perché, tenait en son bec un fromage.",
                         "Maître Renard, par l’odeur alléché, lui tint à peu près ce langage :",
                         "Et bonjour, Monsieur du Corbeau, que vous êtes joli ! que vous me semblez beau !",
                         "Sans mentir, si votre ramage se rapporte à votre plumage, vous êtes le Phénix des hôtes de ces bois.",
                         "À ces mots le Corbeau ne se sent pas de joie, et pour montrer sa belle voix, il ouvre un large bec, laisse tomber sa proie.",
                         "Le Renard s’en saisit, et dit : Mon bon Monsieur, apprenez que tout flatteur vit aux dépens de celui qui l’écoute.",
                         "Cette leçon vaut bien un fromage sans doute.",
                         "Le Corbeau honteux et confus jura, mais un peu tard, qu’on ne l’y prendrait plus."],
               filename=f"audios/Le_corbeau_et_le_renard_{version}.wav",
               device=exec_device,
               language="fr",
               speaker_reference=None)


def the_raven(version, model_id="Meta", exec_device="cpu"):
    os.makedirs("audios", exist_ok=True)

    read_texts(model_id=model_id,
               sentence=['Once upon a midnight dreary, while I pondered, weak, and weary,',
                         'Over many a quaint, and curious volume of forgotten lore,',
                         'While I nodded, nearly napping, suddenly, there came a tapping,',
                         'As of someone gently rapping, rapping at my chamber door.',
                         'Tis some visitor, I muttered, tapping at my chamber door,',
                         'Only this, and nothing more.',
                         'Ah, distinctly, I remember, it was in, the bleak December,',
                         'And each separate dying ember, wrought its ghost upon the floor.',
                         'Eagerly, I wished the morrow, vainly, I had sought to borrow',
                         'From my books surcease of sorrow, sorrow, for the lost Lenore,',
                         'For the rare and radiant maiden, whom the angels name Lenore,',
                         'Nameless here, for evermore.',
                         'And the silken, sad, uncertain, rustling of each purple curtain',
                         'Thrilled me, filled me, with fantastic terrors, never felt before,',
                         'So that now, to still the beating of my heart, I stood repeating',
                         'Tis some visitor, entreating entrance, at my chamber door,',
                         'Some late visitor, entreating entrance, at my chamber door.',
                         'This it is, and nothing more.',
                         'Presently, my soul grew stronger, hesitating then no longer,',
                         'Sir, said I, or Madam, truly, your forgiveness I implore,',
                         'But the fact is, I was napping, and so gently, you came rapping,',
                         'And so faintly, you came tapping, tapping at my chamber door,',
                         'That I scarce was sure I heard you, here, I opened wide the door,',
                         'Darkness there, and nothing more.',
                         'Deep into that darkness, peering, long I stood there wondering, fearing,',
                         'Doubting, dreaming, dreams no mortal ever dared to dream before,',
                         'But the silence was unbroken, and the stillness, gave no token,',
                         'And the only word there spoken, was the whispered word, Lenore?',
                         'This I whispered, and an echo murmured back the word, Lenore!,',
                         'Merely this, and nothing more.',
                         'Back into the chamber turning, all my soul within me burning,',
                         'Soon again, I heard a tapping, somewhat louder than before.',
                         'Surely, said I, surely that is something at my window lattice,',
                         'Let me see, then, what thereat is, and this mystery explore, ',
                         'Let my heart be still a moment, and this mystery explore, ',
                         'Tis the wind, and nothing more!',
                         'Open here I flung the shutter, when, with many a flirt and flutter,',
                         'In there stepped a stately Raven, of the saintly days of yore,',
                         'Not the least obeisance made he, not a minute stopped or stayed he,',
                         'But, with mien of lord or lady, perched above my chamber door, ',
                         'Perched upon a bust of Pallas, just above my chamber door, ',
                         'Perched, and sat, and nothing more.',
                         'Then, this ebony bird beguiling, my sad fancy into smiling,',
                         'By the grave, and stern decorum, of the countenance it wore,',
                         'Though thy crest be shorn and shaven, thou, I said, art sure no craven,',
                         'Ghastly grim, and ancient Raven, wandering from the Nightly shore, ',
                         'Tell me, what thy lordly name is, on the Night’s Plutonian shore!',
                         'Quoth the Raven, Nevermore.'],
               filename=f"audios/the_raven_{version}.wav",
               device=exec_device,
               language="en",
               speaker_reference=None)


if __name__ == '__main__':
    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running on {exec_device}")

    le_corbeau_et_le_renard(version="NEB_vanilla_v1", model_id="NEB", exec_device=exec_device)
