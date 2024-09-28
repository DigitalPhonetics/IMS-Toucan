import os

import torch

from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface


def read_texts(sentence, filename, model_id=None, device="cpu", language="eng", speaker_reference=None, duration_scaling_factor=1.0):
    tts = ToucanTTSInterface(device=device, tts_model_path=model_id)
    tts.set_language(language)
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if type(sentence) == str:
        sentence = [sentence]
    tts.read_to_file(text_list=sentence, file_location=filename, duration_scaling_factor=duration_scaling_factor, prosody_creativity=0.0)
    del tts


def english_test(version, model_id=None, exec_device="cpu", speaker_reference=None):
    os.makedirs("audios", exist_ok=True)

    read_texts(model_id=model_id,
               sentence=["""Once upon a midnight dreary, while I pondered, weak, and weary,
                            Over many a quaint, and curious volume, of forgotten lore,
                            While I nodded, nearly napping, suddenly, there came a tapping,
                            As of someone gently rapping, rapping at my chamber door."""],
               filename=f"audios/{model_id}_english_test_{version}.wav",
               device=exec_device,
               language="eng",
               speaker_reference=speaker_reference)


def japanese_test(version, model_id=None, exec_device="cpu", speaker_reference=None):
    os.makedirs("audios", exist_ok=True)

    read_texts(model_id=model_id,
               sentence=["医師会がなくても、近隣の病院なら紹介してくれると思います。"],
               filename=f"audios/{model_id}_japanese_test_{version}.wav",
               device=exec_device,
               language="jpn",
               speaker_reference=speaker_reference)


def chinese_test(version, model_id=None, exec_device="cpu", speaker_reference=None):
    os.makedirs("audios", exist_ok=True)

    read_texts(model_id=model_id,
               sentence=["李绅 《悯农》 锄禾日当午， 汗滴禾下土。 谁知盘中餐， 粒粒皆辛苦。"],
               filename=f"audios/{model_id}_chinese_test_{version}.wav",
               device=exec_device,
               language="cmn",
               speaker_reference=speaker_reference)


def german_test(version, model_id=None, exec_device="cpu", speaker_reference=None):
    os.makedirs("audios", exist_ok=True)

    read_texts(model_id=model_id,
               sentence=["""Fest gemauert in der Erden,
                            Steht die Form, aus Lehm gebrannt.
                            Heute muss die Glocke werden!
                            Frisch, Gesellen, seid zur Hand!"""],
               filename=f"audios/{model_id}_german_test_{version}.wav",
               device=exec_device,
               language="deu",
               speaker_reference=speaker_reference)


def vietnamese_test(version, model_id=None, exec_device="cpu", speaker_reference=None):
    os.makedirs("audios", exist_ok=True)

    read_texts(model_id=model_id,
               sentence=["""Thân phận,
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
                            không có đất cho nghĩa tự do."""],
               filename=f"audios/{model_id}_vietnamese_test_{version}.wav",
               device=exec_device,
               language="vie",
               speaker_reference=speaker_reference)


def french_test(version, model_id=None, exec_device="cpu", speaker_reference=None):
    os.makedirs("audios", exist_ok=True)

    read_texts(model_id=model_id,
               sentence=["""Maître corbeau, sur un arbre perché,
                            Tenait en son bec un fromage.
                            Maître renard par l'odeur alléché ,
                            Lui tint à peu près ce langage :
                            «Et bonjour Monsieur du Corbeau.
                            Que vous ętes joli! que vous me semblez beau!"""],
               filename=f"audios/{model_id}_french_test_{version}.wav",
               device=exec_device,
               language="fra",
               speaker_reference=speaker_reference)


def all_test(version, model_id=None, exec_device="cpu", speaker_reference=None):
    english_test(version, model_id, exec_device, speaker_reference)
    german_test(version, model_id, exec_device, speaker_reference)
    french_test(version, model_id, exec_device, speaker_reference)
    vietnamese_test(version, model_id, exec_device, speaker_reference)
    japanese_test(version, model_id, exec_device, speaker_reference)
    chinese_test(version, model_id, exec_device, speaker_reference)


if __name__ == '__main__':
    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running on {exec_device}")

    os.makedirs("audios/speaker_references/", exist_ok=True)
    merged_speaker_references = ["audios/speaker_references/" + ref for ref in os.listdir("audios/speaker_references/")]

    all_test(version="version_11",
             model_id=None,  # will use the default
             exec_device=exec_device,
             speaker_reference=merged_speaker_references if merged_speaker_references != [] else None)
