import os
import sys

import torch

from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface


def read_texts(model_id, sentence, filename, device="cpu", language="eng", speaker_reference=None, duration_scaling_factor=1.0, architecture="CFM", prosody_creativity=1.0):
    tts = ToucanTTSInterface(device=device, tts_model_path=model_id, architecture=architecture)
    tts.set_language(language)
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if type(sentence) == str:
        sentence = [sentence]
    tts.read_to_file(text_list=sentence, file_location=filename, duration_scaling_factor=duration_scaling_factor,prosody_creativity=prosody_creativity)
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
               speaker_reference=speaker_reference,
               duration_scaling_factor=1.2)

def create_multiple(version, sentence, model_id="Meta", exec_device="cpu", speaker_reference=None, architecture="CFM", prosody_creativity=1.0):
    os.makedirs("audios", exist_ok=True)
    # ["In restless dreams I walked alone, Narrow streets of cobblestone. Beneath the halo of a streetlamp, I turned my collar to the cold and damp,  When my eyes were stabbed, by the flash of a neon light, That split the night. And touched the sound, of silence."],
    file_name = f"audios/{version}_example.wav"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)          
    read_texts(model_id=model_id,
               sentence=sentence,
               filename=file_name,
               device=exec_device,
               language="eng",
               speaker_reference=speaker_reference,
               architecture=architecture,
               duration_scaling_factor=1.1,
               prosody_creativity=prosody_creativity)


if __name__ == '__main__':
    gpu_id = 7
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    # merged_speaker_references = ["audios/speaker_references/" + ref for ref in os.listdir("audios/speaker_references/")]
    """
    sound_of_silence_single_utt(version="CFM_PED",
                                model_id="Libri_Prosody/CFM/pitch_energy_duration",
                                exec_device=exec_device)

    die_glocke(version="CFM_PED",
               model_id="Libri_Prosody/CFM/pitch_energy_duration",
               exec_device=exec_device)

    the_raven(version="CFM_PED",
              model_id="Libri_Prosody/CFM/pitch_energy_duration",
              exec_device=exec_device)
    """
    models = ["DET"]
    temps = [0.4]

    for model in models:
        for temp in temps:
            if model == "DET":
                samples = 1
            else:
                samples = 1
            for sample in range(samples):
                if model == "DET":
                    model_id ="studyDET_epd_c8_l6_k5_d0.2"
                if model == "CFM":
                    model_id ="epd_c8" 
                if model == "NF":
                    model_id = "studyNF_epd_c8_l6_k5_d0.2"
                if model == "RF":
                    model_id = "epd_c8_reflow2"
                    
                #sentences = ["Galleries are free on thursdays,",
                #              "Jessie dunked the basketball in the hoop,"]
                sentences = []
                
                model_id = f"{model}/{model_id}"
                for i, sentence in enumerate(sentences):

                    create_multiple(version=f"StudyCompletedet/{model}-speaker1-sentence{i}-temp{temp}_{sample}",
                            sentence=sentence,
                            model_id=model_id,
                            exec_device=device,
                            architecture=model,
                            speaker_reference="audios/Study/Human/male.wav",
                            prosody_creativity=temp)
                sentences=["Builders put scaffolding around the windows,",
                           "Michael drinks his tea with milk."]
                sentences=["Michael drinks his tea with milk."]
                for i, sentence in enumerate(sentences):
                    create_multiple(version=f"StudyCompletedet/{model}-speaker2-sentence{i}-temp{temp}_{sample}",
                            sentence=sentence,
                            model_id=model_id,
                            exec_device=device,
                            architecture=model,
                            speaker_reference="audios/Study/Human/female.wav",
                            prosody_creativity=temp)

                    