import os

import torch

from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface


def the_raven_and_the_fox(version, model_id="Meta", exec_device="cpu", speaker_reference=None, vocoder_model_path=None, biggan=False, word_emb_extractor=None):
    os.makedirs("audios", exist_ok=True)
    os.makedirs(f"audios/{version}", exist_ok=True)
    tts = ToucanTTSInterface(device=exec_device, tts_model_path=model_id, vocoder_model_path=vocoder_model_path, faster_vocoder=not biggan, word_emb_extractor=word_emb_extractor)
    tts.set_language("en")
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)

    for i, sentence in enumerate(["Master Raven, on a perched tree, was holding a cheese in his beak.",
                                  "Master Fox, attracted by the smell, told him more or less this language:",
                                  "And hello, Master Raven, how pretty you are! How beautiful you seem to me!",
                                  "No lie, if your bearing is anything like your plumage, you are the Phoenix of the hosts in these woods.",
                                  "At these words the Raven does not feel happy, and to show his beautiful voice, he opens a wide beak, drops his prey.",
                                  "The Fox seized it, and said: My good Sir, learn that every flatterer lives at the expense of the one who listens to him.",
                                  "This lesson is worth a cheese without doubt.",
                                  "The ashamed and confused Raven swore, but a little late, that he would not be caught again.",
                                  "Master Raven, on a perched tree, was holding a cheese in his beak. Master Fox, attracted by the smell, told him more or less this language: And hello, Master Raven, how pretty you are! How beautiful you seem to me! No lie, if your bearing is anything like your plumage, you are the Phoenix of the hosts in these woods. At these words the Raven does not feel happy, and to show his beautiful voice, he opens a wide beak, drops his prey. The Fox seized it, and said: My good Sir, learn that every flatterer lives at the expense of the one who listens to him. This lesson is worth a cheese without doubt. The ashamed and confused Raven swore, but a little late, that he would not be caught again."
                                  ]):
        tts.read_to_file(text_list=[sentence], file_location=f"audios/{version}/The_raven_and_the_fox_{i}.wav")

def poem(version, model_id="Meta", exec_device="cpu", speaker_reference=None, vocoder_model_path=None, biggan=False, word_emb_extractor=None):
    os.makedirs("audios", exist_ok=True)
    os.makedirs(f"audios/{version}", exist_ok=True)
    tts = ToucanTTSInterface(device=exec_device, tts_model_path=model_id, vocoder_model_path=vocoder_model_path, faster_vocoder=not biggan, word_emb_extractor=word_emb_extractor)
    tts.set_language("en")
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)

    for i, sentence in enumerate(['Once upon a midnight dreary, while I pondered, weak, and weary,',
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
                                'Thrilled me, filled me, with fantastic terrors, never felt before.']):
        tts.read_to_file(text_list=[sentence], file_location=f"audios/{version}/Poem_{i}.wav")

def test_sentence(version, model_id="Meta", exec_device="cpu", speaker_reference=None, vocoder_model_path=None, biggan=False, word_emb_extractor=None):
    os.makedirs("audios", exist_ok=True)
    os.makedirs(f"audios/{version}", exist_ok=True)
    tts = ToucanTTSInterface(device=exec_device, tts_model_path=model_id, vocoder_model_path=vocoder_model_path, faster_vocoder=not biggan, word_emb_extractor=word_emb_extractor)
    tts.set_language("en")
    #sentence = "Well, she said, if I had had your bringing up I might have had as good a temper as you, but now I don't believe I ever shall."
    sentence = "Did he drive a red car to work? No he drove a blue car to work."
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    tts.read_to_file(text_list=[sentence], file_location=f"audios/{version}/test_sentence.wav")

def test_controllable(version, model_id="Meta", exec_device="cpu", speaker_reference=None, vocoder_model_path=None, biggan=False, word_emb_extractor=None):
    os.makedirs("audios", exist_ok=True)
    os.makedirs(f"audios/{version}", exist_ok=True)
    tts = ToucanTTSInterface(device=exec_device, tts_model_path=model_id, vocoder_model_path=vocoder_model_path, faster_vocoder=not biggan, word_emb_extractor=word_emb_extractor)
    tts.set_language("en")
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)

    for i, sentence in enumerate(['I am so happy to see you!',
                                  'Today is a beautiful day and the sun is shining.',
                                  'He seemed to be quite lucky as he was smiling at me.',
                                  'She laughed and said: This is so funny.',
                                  'No, this is horrible!',
                                  'I am so sad, why is this so depressing?',
                                  'Be careful!, Cried the woman',
                                  'This makes me feel bad.']):
        tts.read_to_file(text_list=[sentence], file_location=f"audios/{version}/Controllable_{i}.wav")


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"8"
    exec_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    #exec_device = "cpu"
    print(f"running on {exec_device}")

    use_speaker_reference = False
    use_word_emb = True

    if use_word_emb:
        from Preprocessing.word_embeddings.BERTWordEmbeddingExtractor import BERTWordEmbeddingExtractor
        word_embedding_extractor = BERTWordEmbeddingExtractor()
    else:
        word_embedding_extractor = None

    if use_speaker_reference:
        #speaker_reference = "/mount/resources/speech/corpora/Blizzard2013/train/segmented/wavn/CA-BB-05-19.wav"
        speaker_reference = "/mount/resources/speech/corpora/LibriTTS/all_clean/1638/84448/1638_84448_000057_000006.wav"
    else:
        speaker_reference = None

    test_sentence(version="ToucanTTS_02_Blizzard2013_word_emb_bert", model_id="02_Blizzard2013_word_emb_bert", exec_device=exec_device, vocoder_model_path=None, biggan=True, speaker_reference=speaker_reference, word_emb_extractor=word_embedding_extractor)
