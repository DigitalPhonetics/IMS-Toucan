import os

import torch

from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface


def read_texts(model_id, sentence, filename, device="cpu", language="en", speaker_reference=None, faster_vocoder=False):
    tts = ToucanTTSInterface(device=device, tts_model_path=model_id, faster_vocoder=faster_vocoder)
    tts.set_language(language)
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if type(sentence) == str:
        sentence = [sentence]
    tts.read_to_file(text_list=sentence, file_location=filename)
    del tts


def the_raven(version, model_id="Meta", exec_device="cpu", speed_over_quality=True, speaker_reference=None):
    os.makedirs("audios", exist_ok=True)

    read_texts(model_id=model_id,
               sentence=['Once upon a midnight dreary, while I pondered, weak, and weary,',
                         'Over many a quaint, and curious volume of forgotten lore,',
                         'While I nodded, nearly napping, suddenly, there came a tapping,',
                         'As of someone gently rapping, rapping at my chamber door.',
                         'Tis some visitor, I muttered, tapping at my chamber door,',
                         'Only this, and nothing more.',
                         'Ah, distinctly, I remember, it was in the bleak December,',
                         'And each separate dying ember, wrought its ghost upon the floor.',
                         'Eagerly, I wished the morrow, vainly, I had sought to borrow',
                         'From my books surcease of sorrow, sorrow, for the lost Lenore,',
                         'For the rare and radiant maiden, whom the angels name Lenore,',
                         'Nameless here, for evermore.',
                         'And the silken, sad, uncertain, rustling of each purple curtain',
                         'Thrilled me, filled me, with fantastic terrors, never felt before.'],
               filename=f"audios/the_raven_{version}.wav",
               device=exec_device,
               language="en",
               speaker_reference=speaker_reference,
               faster_vocoder=speed_over_quality)


if __name__ == '__main__':
    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running on {exec_device}")

    the_raven(version="MetaBaseline",
              model_id="Meta",
              exec_device=exec_device,
              speed_over_quality=exec_device != "cuda")
