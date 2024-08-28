import os

import torch

from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface


def read_texts(model_id, sentence, filename, device="cpu", language="eng", speaker_reference=None, duration_scaling_factor=1.0):
    tts = ToucanTTSInterface(device=device, tts_model_path=model_id)
    tts.set_language(language)
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if type(sentence) == str:
        sentence = [sentence]
    tts.read_to_file(text_list=sentence, file_location=filename, duration_scaling_factor=duration_scaling_factor, prosody_creativity=0.0)
    del tts


def new_test(version, lang, texts, filename, model_id="Meta", exec_device="cpu", speaker_reference=None):
    os.makedirs("audios", exist_ok=True)
    read_texts(model_id=model_id,
               sentence=[texts],
               filename=f"audios/{model_id}_{filename}_{version}.wav",
               device=exec_device,
               language=lang,
               speaker_reference=speaker_reference)


def all_test(version, model_id="Meta", exec_device="cpu", speaker_reference=None):
    new_test(version,
             lang="eng",
             model_id=model_id,
             texts="""Once upon a midnight dreary, while I pondered, weak, and weary,
                      Over many a quaint, and curious volume, of forgotten lore,
                      While I nodded, nearly napping, suddenly, there came a tapping,
                      As of someone gently rapping, rapping at my chamber door.""",
             filename="english_test",
             exec_device=exec_device,
             speaker_reference=speaker_reference)
    new_test(version,
             lang="deu",
             model_id=model_id,
             texts="""Fest gemauert in der Erden,
                      Steht die Form, aus Lehm gebrannt.
                      Heute muss die Glocke werden!
                      Frisch, Gesellen, seid zur Hand!""",
             filename="german_test",
             exec_device=exec_device,
             speaker_reference=speaker_reference)
    new_test(version,
             lang="fra",
             model_id=model_id,
             texts="""Maître corbeau, sur un arbre perché,
                      Tenait en son bec un fromage.
                      Maître renard par l'odeur alléché ,
                      Lui tint à peu près ce langage :
                      «Et bonjour Monsieur du Corbeau.
                      Que vous ętes joli! que vous me semblez beau!""",
             filename="french_test",
             exec_device=exec_device,
             speaker_reference=speaker_reference)
    new_test(version,
             lang="vie",
             model_id=model_id,
             texts="""Thân phận,
                      ở một nơi luôn phải nhắc mình,
                      im miệng,
                      thân phận,
                      là khi nói về quá khứ,
                      ngó trước nhìn sau,
                      là phải biết nhắm mắt bịt tai làm lơ,
                      thờ ơ,
                      với tất cả những điều gai chướng,
                      thân phận chúng tôi ở đó,
                      những quyển sách chuyền tay nhau như ăn cắp,
                      ngôn luận ư?
                      không có đất cho nghĩa tự do.""",
             filename="vietnamese_test",
             exec_device=exec_device,
             speaker_reference=speaker_reference)
    new_test(version,
             lang="jpn",
             model_id=model_id,
             texts="医師会がなくても、近隣の病院なら紹介してくれると思います。",
             filename="japanese_test",
             exec_device=exec_device,
             speaker_reference=speaker_reference)
    new_test(version,
             lang="cmn",
             model_id=model_id,
             texts="李绅 《悯农》 锄禾日当午， 汗滴禾下土。 谁知盘中餐， 粒粒皆辛苦。",
             filename="chinese_test",
             exec_device=exec_device,
             speaker_reference=speaker_reference)


if __name__ == '__main__':
    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running on {exec_device}")

    merged_speaker_references = ["audios/speaker_references/" + ref for ref in os.listdir("audios/speaker_references/")]

    all_test(version="version_11",
             model_id="Meta",
             exec_device=exec_device,
             speaker_reference=merged_speaker_references if merged_speaker_references != [] else None)
