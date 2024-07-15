import os

import torch

from Architectures.ControllabilityGAN.GAN import GanWrapper
from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface
from Utility.storage_config import MODELS_DIR


class ControllableInterface:

    def __init__(self, gpu_id="cpu", available_artificial_voices=1000):
        if gpu_id == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        else:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
        self.device = "cuda" if gpu_id != "cpu" else "cpu"
        self.model = ToucanTTSInterface(device=self.device, tts_model_path="Meta")
        self.wgan = GanWrapper(os.path.join(MODELS_DIR, "Embedding", "embedding_gan.pt"), device=self.device)
        self.generated_speaker_embeds = list()
        self.available_artificial_voices = available_artificial_voices
        self.current_language = ""
        self.current_accent = ""

    def read(self,
             prompt,
             reference_audio,
             language,
             accent,
             voice_seed,
             prosody_creativity,
             duration_scaling_factor,
             pause_duration_scaling_factor,
             pitch_variance_scale,
             energy_variance_scale,
             emb_slider_1,
             emb_slider_2,
             emb_slider_3,
             emb_slider_4,
             emb_slider_5,
             emb_slider_6,
             loudness_in_db
             ):
        if self.current_language != language:
            self.model.set_phonemizer_language(language)
            print(f"switched phonemizer language to {language}")
            self.current_language = language
        if self.current_accent != accent:
            self.model.set_accent_language(accent)
            print(f"switched accent language to {accent}")
            self.current_accent = accent
        if reference_audio is None:
            self.wgan.set_latent(voice_seed)
            controllability_vector = torch.tensor([emb_slider_1,
                                                   emb_slider_2,
                                                   emb_slider_3,
                                                   emb_slider_4,
                                                   emb_slider_5,
                                                   emb_slider_6], dtype=torch.float32)
            embedding = self.wgan.modify_embed(controllability_vector)
            self.model.set_utterance_embedding(embedding=embedding)
        else:
            self.model.set_utterance_embedding(reference_audio)

        phones = self.model.text2phone.get_phone_string(prompt)
        if len(phones) > 1800:
            if language == "deu":
                prompt = "Deine Eingabe war zu lang. Bitte versuche es entweder mit einem kürzeren Text oder teile ihn in mehrere Teile auf."
            elif language == "ell":
                prompt = "Η εισήγησή σας ήταν πολύ μεγάλη. Παρακαλώ δοκιμάστε είτε ένα μικρότερο κείμενο είτε χωρίστε το σε διάφορα μέρη."
            elif language == "spa":
                prompt = "Su entrada es demasiado larga. Por favor, intente un texto más corto o divídalo en varias partes."
            elif language == "fin":
                prompt = "Vastauksesi oli liian pitkä. Kokeile joko lyhyempää tekstiä tai jaa se useampaan osaan."
            elif language == "rus":
                prompt = "Ваш текст слишком длинный. Пожалуйста, попробуйте либо сократить текст, либо разделить его на несколько частей."
            elif language == "hun":
                prompt = "Túl hosszú volt a bevitele. Kérjük, próbáljon meg rövidebb szöveget írni, vagy ossza több részre."
            elif language == "nld":
                prompt = "Uw input was te lang. Probeer een kortere tekst of splits het in verschillende delen."
            elif language == "fra":
                prompt = "Votre saisie était trop longue. Veuillez essayer un texte plus court ou le diviser en plusieurs parties."
            elif language == 'pol':
                prompt = "Twój wpis był zbyt długi. Spróbuj skrócić tekst lub podzielić go na kilka części."
            elif language == 'por':
                prompt = "O seu contributo foi demasiado longo. Por favor, tente um texto mais curto ou divida-o em várias partes."
            elif language == 'ita':
                prompt = "Il tuo input era troppo lungo. Per favore, prova un testo più corto o dividilo in più parti."
            elif language == 'cmn':
                prompt = "你的输入太长了。请尝试使用较短的文本或将其拆分为多个部分。"
            elif language == 'vie':
                prompt = "Đầu vào của bạn quá dài. Vui lòng thử một văn bản ngắn hơn hoặc chia nó thành nhiều phần."
            else:
                prompt = "Your input was too long. Please try either a shorter text or split it into several parts."
                if self.current_language != "eng":
                    self.model.set_phonemizer_language("eng")
                    self.current_language = "eng"
                if self.current_accent != "eng":
                    self.model.set_accent_language("eng")
                    self.current_accent = "eng"

        print(prompt + "\n\n")
        wav, sr, fig = self.model(prompt,
                                  input_is_phones=False,
                                  duration_scaling_factor=duration_scaling_factor,
                                  pitch_variance_scale=pitch_variance_scale,
                                  energy_variance_scale=energy_variance_scale,
                                  pause_duration_scaling_factor=pause_duration_scaling_factor,
                                  return_plot_as_filepath=True,
                                  prosody_creativity=prosody_creativity,
                                  loudness_in_db=loudness_in_db)
        return sr, wav, fig
