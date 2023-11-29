import os

import torch

from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface


def read_texts(model_id, sentence, filename, device="cpu", language="eng", speaker_reference=None, pause_duration_scaling_factor=1.0):
    tts = ToucanTTSInterface(device=device, tts_model_path=model_id)
    tts.set_language(language)
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if type(sentence) == str:
        sentence = [sentence]
    tts.read_to_file(text_list=sentence, file_location=filename, pause_duration_scaling_factor=pause_duration_scaling_factor)
    del tts


def the_raven(version, model_id="Meta", exec_device="cpu", speaker_reference=None):
    os.makedirs("audios", exist_ok=True)

    read_texts(model_id=model_id,
               sentence=['Once upon a midnight dreary, while I pondered, weak, and weary,',
                         'Over many a quaint, and curious volume, of forgotten lore,',
                         'While I nodded, nearly napping, suddenly, there came a tapping,',
                         'As of someone gently rapping, rapping at my chamber door.',
                         'Ah, distinctly, I remember, it was in the bleak December,',
                         'And each separate dying ember, wrought its ghost upon the floor.',
                         'Eagerly, I wished the morrow, vainly, I had sought to borrow',
                         'From my books surcease of sorrow, sorrow, for the lost Lenore,',
                         'And the silken, sad, uncertain, rustling of each purple curtain',
                         'Thrilled me, filled me, with fantastic terrors, never felt before.'],
               filename=f"audios/the_raven_{version}.wav",
               device=exec_device,
               language="eng",
               speaker_reference=speaker_reference)


def sound_of_silence_single_utt(version, model_id="Meta", exec_device="cpu", speaker_reference=None):
    os.makedirs("audios", exist_ok=True)

    read_texts(model_id=model_id,
               sentence=["""Hello darkness, my old friend.
I′ve come to talk with you again.
Because a vision softly creeping,
Left its seeds while I was sleeping,
And the vision, that was planted, in my brain,
Still remains,
Within the sound, of silence.

In restless dreams I walked alone,
Narrow streets of cobblestone.
Beneath the halo of a streetlamp,
I turned my collar to the cold and damp,
When my eyes were stabbed, by the flash of a neon light,
That split the night.
And touched the sound, of silence.


And in the naked light I saw,
Ten thousand people, maybe more.
People talking without speaking.
People hearing without listening.
People writing songs, that voices never shared.
No one dared,
Disturb the sound of silence.

"Fools", said I, "You do not know.
Silence, like a cancer grows.
Hear my words that I might teach you,
Take my arms that I might reach you!"
But my words, like silent raindrops fell,
And echoed, in the wells, of silence.

And the people bowed and prayed,
To the neon god they made.
And the sign flashed out its warning,
In the words that it was forming,
And the sign said, "The words of the prophets, are written on the subway walls.
In tenement halls."
And whispered, in the sounds, of silence."""],
               filename=f"audios/sound_of_silence_as_single_utterance_{version}.wav",
               device=exec_device,
               language="eng",
               speaker_reference=speaker_reference,
               pause_duration_scaling_factor=2.0)


def die_glocke(version, model_id="Meta", exec_device="cpu", speaker_reference=None):
    os.makedirs("audios", exist_ok=True)

    read_texts(model_id=model_id,
               sentence=['Fest gemauert in der Erden,',
                         'Steht die Form, aus Lehm gebrannt.',
                         'Heute muss die Glocke werden!',
                         'Frisch, Gesellen, seid zur Hand!'],
               filename=f"audios/die_glocke_{version}.wav",
               device=exec_device,
               language="deu",
               speaker_reference=speaker_reference)


if __name__ == '__main__':
    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running on {exec_device}")

    sound_of_silence_single_utt(version="BigModel",
                                model_id="Nancy",
                                exec_device=exec_device,
                                speaker_reference="audios/reference_audios/german_female.wav")
