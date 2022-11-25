import os

import torch

from InferenceInterfaces.Controllability.GAN import GanWrapper
from InferenceInterfaces.PortaSpeechInterface import PortaSpeechInterface


class ControllableInterface:

    def __init__(self, gpu_id="cpu"):
        if gpu_id == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        else:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = PortaSpeechInterface(device=self.device, tts_model_path="Meta")
        self.wgan = GanWrapper('Models/Embedding/embedding_gan.pt', device=self.device)
        self.current_language = "English"
        self.current_accent = "English"
        self.language_id_lookup = {
            "English":    "en",
            "German":     "de",
            "Greek":      "el",
            "Spanish":    "es",
            "Finnish":    "fi",
            "Russian":    "ru",
            "Hungarian":  "hu",
            "Dutch":      "nl",
            "French":     "fr",
            'Polish':     "pl",
            'Portuguese': "pt",
            'Italian':    "it",
            'Chinese':    "cmn",
            'Vietnamese': "vi",
        }

    def read(self,
             prompt,
             language,
             accent,
             voice_seed,
             duration_scaling_factor,
             pause_duration_scaling_factor,
             pitch_variance_scale,
             energy_variance_scale,
             emb_slider_1,
             emb_slider_2,
             emb_slider_6,
             ):
        language = language.split()[0]
        accent = accent.split()[0]
        if self.current_language != language:
            self.model.set_phonemizer_language(self.language_id_lookup[language])
            self.current_language = language
        if self.current_accent != accent:
            self.model.set_accent_language(self.language_id_lookup[accent])
            self.current_accent = accent

        self.wgan.set_latent(voice_seed)

        controllability_vector = torch.tensor(
            [emb_slider_1, emb_slider_2, 0.0, 0.0, 0.0, emb_slider_6], dtype=torch.float32)
        embedding = self.wgan.modify_embed(controllability_vector)
        self.model.set_utterance_embedding(embedding=embedding)

        phones = self.model.text2phone.get_phone_string(prompt)
        if len(phones) > 1800:
            if language == "English":
                prompt = "Your input was too long. Please try either a shorter text or split it into several parts."
            elif language == "German":
                prompt = "Deine Eingabe war zu lang. Bitte versuche es entweder mit einem kürzeren Text oder teile ihn in mehrere Teile auf."
            elif language == "Greek":
                prompt = "Η εισήγησή σας ήταν πολύ μεγάλη. Παρακαλώ δοκιμάστε είτε ένα μικρότερο κείμενο είτε χωρίστε το σε διάφορα μέρη."
            elif language == "Spanish":
                prompt = "Su entrada es demasiado larga. Por favor, intente un texto más corto o divídalo en varias partes."
            elif language == "Finnish":
                prompt = "Vastauksesi oli liian pitkä. Kokeile joko lyhyempää tekstiä tai jaa se useampaan osaan."
            elif language == "Russian":
                prompt = "Ваш текст слишком длинный. Пожалуйста, попробуйте либо сократить текст, либо разделить его на несколько частей."
            elif language == "Hungarian":
                prompt = "Túl hosszú volt a bevitele. Kérjük, próbáljon meg rövidebb szöveget írni, vagy ossza több részre."
            elif language == "Dutch":
                prompt = "Uw input was te lang. Probeer een kortere tekst of splits het in verschillende delen."
            elif language == "French":
                prompt = "Votre saisie était trop longue. Veuillez essayer un texte plus court ou le diviser en plusieurs parties."
            elif language == 'Polish':
                prompt = "Twój wpis był zbyt długi. Spróbuj skrócić tekst lub podzielić go na kilka części."
            elif language == 'Portuguese':
                prompt = "O seu contributo foi demasiado longo. Por favor, tente um texto mais curto ou divida-o em várias partes."
            elif language == 'Italian':
                prompt = "Il tuo input era troppo lungo. Per favore, prova un testo più corto o dividilo in più parti."
            elif language == 'Chinese':
                prompt = "你的输入太长了。请尝试使用较短的文本或将其拆分为多个部分。"
            elif language == 'Vietnamese':
                prompt = "Đầu vào của bạn quá dài. Vui lòng thử một văn bản ngắn hơn hoặc chia nó thành nhiều phần."
            phones = self.model.text2phone.get_phone_string(prompt)

        wav = self.model(phones,
                         input_is_phones=True,
                         duration_scaling_factor=duration_scaling_factor,
                         pitch_variance_scale=pitch_variance_scale,
                         energy_variance_scale=energy_variance_scale,
                         pause_duration_scaling_factor=pause_duration_scaling_factor)
        return 24000, wav
