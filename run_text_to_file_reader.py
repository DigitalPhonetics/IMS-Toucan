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


def le_corbeau_et_le_renard(version, model_id="Meta", exec_device="cpu"):
    os.makedirs("audios", exist_ok=True)

    read_texts(model_id=model_id,
               sentence=["Maître Corbeau, sur un arbre perché, tenait en son bec un fromage.",
                         "Maître Renard, par l’odeur alléché, lui tint à peu près ce langage:",
                         "Et bonjour, Monsieur du Corbeau, que vous êtes joli! que vous me semblez beau!",
                         "Sans mentir, si votre ramage se rapporte à votre plumage, vous êtes le Phénix des hôtes de ces bois.",
                         "À ces mots le Corbeau ne se sent pas de joie, et pour montrer sa belle voix, il ouvre un large bec, laisse tomber sa proie.",
                         "Le Renard s’en saisit, et dit: Mon bon Monsieur, apprenez que tout flatteur vit aux dépens de celui qui l’écoute.",
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
                         'Thrilled me, filled me, with fantastic terrors, never felt before'],
               filename=f"audios/the_raven_{version}.wav",
               device=exec_device,
               language="en",
               speaker_reference=None)


if __name__ == '__main__':
    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running on {exec_device}")

    the_raven(version="default_model", model_id="Meta", exec_device=exec_device)
