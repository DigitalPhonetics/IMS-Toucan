import os

import torch

from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface


def the_raven_and_the_fox(version, model_id="Meta", exec_device="cpu", speaker_reference=None, vocoder_model_path=None, biggan=False, sent_emb_extractor=None, prompt:str=None, word_emb_extractor=None):
    os.makedirs("audios", exist_ok=True)
    os.makedirs(f"audios/{version}", exist_ok=True)
    tts = ToucanTTSInterface(device=exec_device, tts_model_path=model_id, vocoder_model_path=vocoder_model_path, faster_vocoder=not biggan, sent_emb_extractor=sent_emb_extractor, word_emb_extractor=word_emb_extractor)
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
        if prompt is not None:
            tts.set_sentence_embedding(prompt)
        tts.read_to_file(text_list=[sentence], file_location=f"audios/{version}/The_raven_and_the_fox_{i}.wav")

def poem(version, model_id="Meta", exec_device="cpu", speaker_reference=None, vocoder_model_path=None, biggan=False, sent_emb_extractor=None, prompt:str=None, word_emb_extractor=None):
    os.makedirs("audios", exist_ok=True)
    os.makedirs(f"audios/{version}", exist_ok=True)
    tts = ToucanTTSInterface(device=exec_device, tts_model_path=model_id, vocoder_model_path=vocoder_model_path, faster_vocoder=not biggan, sent_emb_extractor=sent_emb_extractor, word_emb_extractor=word_emb_extractor)
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
        if prompt is not None:
            tts.set_sentence_embedding(prompt)
        tts.read_to_file(text_list=[sentence], file_location=f"audios/{version}/Poem_{i}.wav")

def test_sentence(version, model_id="Meta", exec_device="cpu", speaker_reference=None, vocoder_model_path=None, biggan=False, sent_emb_extractor=None, prompt:str=None, word_emb_extractor=None):
    os.makedirs("audios", exist_ok=True)
    os.makedirs(f"audios/{version}", exist_ok=True)
    tts = ToucanTTSInterface(device=exec_device, tts_model_path=model_id, vocoder_model_path=vocoder_model_path, faster_vocoder=not biggan, sent_emb_extractor=sent_emb_extractor, word_emb_extractor=word_emb_extractor)
    tts.set_language("en")
    sentence = "Well, she said, if I had had your bringing up I might have had as good a temper as you, but now I don't believe I ever shall."
    #sentence = "But yours, your regard was new compared with; Fanny, think of me!"
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if prompt is not None:
        tts.set_sentence_embedding(prompt)
    tts.read_to_file(text_list=[sentence], file_location=f"audios/{version}/test_sentence.wav")

def test_controllable(version, model_id="Meta", exec_device="cpu", speaker_reference=None, vocoder_model_path=None, biggan=False, sent_emb_extractor=None, prompt:str=None, make_preprompt=False, word_emb_extractor=None):
    os.makedirs("audios", exist_ok=True)
    os.makedirs(f"audios/{version}", exist_ok=True)
    tts = ToucanTTSInterface(device=exec_device, tts_model_path=model_id, vocoder_model_path=vocoder_model_path, faster_vocoder=not biggan, sent_emb_extractor=sent_emb_extractor, word_emb_extractor=word_emb_extractor)
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
        if prompt is not None:
            if make_preprompt:
                prompt = prompt + ' ' + sentence
            tts.set_sentence_embedding(prompt)
        tts.read_to_file(text_list=[sentence], file_location=f"audios/{version}/Controllable_{i}.wav")

def test_promptspeech(version, model_id="Meta", exec_device="cpu", speaker_reference=None, vocoder_model_path=None, biggan=False, sent_emb_extractor=None, prompt:str=None, make_preprompt=False):
    os.makedirs("audios", exist_ok=True)
    os.makedirs(f"audios/{version}", exist_ok=True)
    tts = ToucanTTSInterface(device=exec_device, tts_model_path=model_id, vocoder_model_path=vocoder_model_path, faster_vocoder=not biggan, sent_emb_extractor=sent_emb_extractor)
    tts.set_language("en")
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    
    prompts = ["Women's voice reminds whispers to be fast and high tone",
               "Lady loud but bass tone saying",
               "A girl shouted in a low speed",
               "Seek a loud female voice with a treble, fast speed",
               "A male bass voice said lowly and quietly",
               "Minor men whisper and fast, talk with high tone",
               "The male voice loud and high tone and the speed is slow",
               "His speaking rate is rapidly and loudly"]

    for i, sentence in enumerate(['The next day I left Marsh End for Morton.',
                                  'It was not, she knew, that night had come, but something as dark as night had come.',
                                  "He immediately answered in Hook's voice:",
                                  'Give me some brandy.',
                                  "Perhaps I'd better quit talking.",
                                  'Up and down the street he went, and in and out the lanes, but no traces of the pig could he find anywhere.',
                                  'There were no windows whatever, and only one or two slight crevices through which the light came.',
                                  'When Wendy returned diffidently she found peter sitting on the bed post crowing gloriously, while Jane in her nighty was flying round the room in solemn ecstasy.']):
        if make_preprompt:
            prompt = prompts[i] + ' ' + sentence
            tts.set_sentence_embedding(prompt)
        else:
            tts.set_sentence_embedding(prompts[i])
        tts.read_to_file(text_list=[sentence], file_location=f"audios/{version}/test_promptspeech_{i}.wav")


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"1,2"
    exec_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    #exec_device = "cpu"
    print(f"running on {exec_device}")

    use_speaker_reference = False
    use_sent_emb = True 
    use_word_emb = True
    use_prompt = False

    if use_sent_emb:
        #import tensorflow
        #gpus = tensorflow.config.experimental.list_physical_devices('GPU')
        #tensorflow.config.experimental.set_visible_devices(gpus[1], 'GPU')
        #from Preprocessing.sentence_embeddings.LEALLASentenceEmbeddingExtractor import LEALLASentenceEmbeddingExtractor as SentenceEmbeddingExtractor

        #from Preprocessing.sentence_embeddings.LASERSentenceEmbeddingExtractor import LASERSentenceEmbeddingExtractor as SentenceEmbeddingExtractor
        #from Preprocessing.sentence_embeddings.STSentenceEmbeddingExtractor import STSentenceEmbeddingExtractor as SentenceEmbeddingExtractor
        from Preprocessing.sentence_embeddings.BERTSentenceEmbeddingExtractor import BERTSentenceEmbeddingExtractor as SentenceEmbeddingExtractor

        #sent_emb_extractor = SentenceEmbeddingExtractor(model="mpnet")
        sent_emb_extractor = SentenceEmbeddingExtractor(pooling="cls")
        #sent_emb_extractor = SentenceEmbeddingExtractor()
    else:
        sent_emb_extractor = None

    if use_word_emb:
        from Preprocessing.word_embeddings.BERTWordEmbeddingExtractor import BERTWordEmbeddingExtractor
        word_embedding_extractor = BERTWordEmbeddingExtractor()
    else:
        word_embedding_extractor = None

    if use_speaker_reference:
        speaker_reference = "/mount/resources/speech/corpora/Blizzard2013/train/segmented/wavn/CA-BB-07-04.wav"
        #speaker_reference = "/mount/resources/speech/corpora/LibriTTS/all_clean/1638/84448/1638_84448_000057_000006.wav"
    else:
        speaker_reference = None

    if use_prompt:
        #prompt = "Well, she said, if I had had your bringing up I might have had as good a temper as you, but now I don't believe I ever shall."
        prompt = "No, this is horrible!"
    else:
        prompt = None

    test_controllable(version="ToucanTTS_03_Blizzard2013_sent_word_emb_a12_loss_bertcls_keep", model_id="03_Blizzard2013_sent_word_emb_a12_loss_bertcls_keep", exec_device=exec_device, vocoder_model_path=None, biggan=True, speaker_reference=speaker_reference, sent_emb_extractor=sent_emb_extractor, word_emb_extractor=word_embedding_extractor, prompt=prompt)