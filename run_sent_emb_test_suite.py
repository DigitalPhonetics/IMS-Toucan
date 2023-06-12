import os

import torch

from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface

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

    sentence = "result in some degree of interference with the personal liberty of those involved."
    tts.read_to_file(text_list=[sentence], file_location=f"audios/{version}/test_sentence.flac", increased_compatibility_mode=True)

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
                                  'I am so angry!',
                                  'What a surprise!',
                                  'This is a neutral test sentence with medium length, which should have relatively neutral prosody, and can be used to test the controllability through textual prompts.']):
        tts.read_to_file(text_list=[sentence], file_location=f"audios/{version}/Controllable_{i}.wav", increased_compatibility_mode=True)


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"2"
    exec_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"running on {exec_device}")

    use_speaker_reference = False
    use_sent_emb = True
    use_word_emb = False
    use_prompt = False
    use_xvect = False
    use_ecapa = False
    use_speaker_id = True

    if use_speaker_id:
        speaker_id = 3 + 1 + 91 + 24
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
        speaker_reference = "/mount/arbeitsdaten/synthesis/bottts/IMS-Toucan/Corpora/EmoVDB/sam/amused_1-28_0010-16bit.wav"
    else:
        speaker_reference = None

    if use_prompt:
        #prompt = "Well, she said, if I had had your bringing up I might have had as good a temper as you, but now I don't believe I ever shall."
        #prompt = "I am so angry!"
        #prompt = "Roar with laughter, this is funny."
        #prompt = "Ew, this is disgusting."
        prompt = "Wow, what a surprise!."
        #prompt = "This is very sad."
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

    test_controllable(version="ToucanTTS_12_EmoMulti_sent_emb_emoBERTcls_static_SE2_each",
                      model_id="12_EmoMulti_sent_emb_emoBERTcls_static_SE2_each",
                      exec_device=exec_device,
                      vocoder_model_path=None,
                      biggan=True,
                      speaker_reference=speaker_reference,
                      sent_emb_extractor=sent_emb_extractor,
                      word_emb_extractor=word_embedding_extractor,
                      prompt=prompt,
                      xvect_model=xvect_model, 
                      speaker_id=speaker_id)
