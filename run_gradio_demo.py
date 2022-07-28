import gradio as gr
import numpy as np
import torch

from InferenceInterfaces.InferenceFastSpeech2 import InferenceFastSpeech2


def float2pcm(sig, dtype='int16'):
    """
    https://gist.github.com/HudsonHuang/fbdf8e9af7993fe2a91620d3fb86a182
    """
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")
    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


class TTS_Interface:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = InferenceFastSpeech2(device=self.device, model_name="EmoGST")
        self.current_language = "English"
        self.current_accent = "English"
        self.language_id_lookup = {
            "English"   : "en",
            "German"    : "de",
            "Greek"     : "el",
            "Spanish"   : "es",
            "Finnish"   : "fi",
            "Russian"   : "ru",
            "Hungarian" : "hu",
            "Dutch"     : "nl",
            "French"    : "fr",
            'Polish'    : "pl",
            'Portuguese': "pt",
            'Italian'   : "it",
            }

    def read(self, prompt, language, accent, emb_slider_1, emb_slider_2, emb_slider_3, emb_slider_4, emb_slider_5, emb_slider_6):
        language = language.split()[0]
        accent = accent.split()[0]
        if self.current_language != language:
            self.model.set_phonemizer_language(self.language_id_lookup[language])
            self.current_language = language
        if self.current_accent != accent:
            self.model.set_accent_language(self.language_id_lookup[accent])
            self.current_accent = accent

        embedding = None  # <-- MODIFY THIS
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
            phones = self.model.text2phone.get_phone_string(prompt)

        wav = self.model(phones)
        return 48000, float2pcm(wav.cpu().numpy())


if __name__ == '__main__':
    meta_model = TTS_Interface()
    iface = gr.Interface(fn=meta_model.read,
                         inputs=[gr.inputs.Textbox(lines=2,
                                                   placeholder="write what you want the synthesis to read here...)",
                                                   label="Text input"),
                                 gr.inputs.Dropdown(['English Text',
                                                     'German Text',
                                                     'Greek Text',
                                                     'Spanish Text',
                                                     'Finnish Text',
                                                     'Russian Text',
                                                     'Hungarian Text',
                                                     'Dutch Text',
                                                     'French Text',
                                                     'Polish Text',
                                                     'Portuguese Text',
                                                     'Italian Text'], type="value", default='English Text', label="Select the Language of the Text"),
                                 gr.inputs.Dropdown(['English Accent',
                                                     'German Accent',
                                                     'Greek Accent',
                                                     'Spanish Accent',
                                                     'Finnish Accent',
                                                     'Russian Accent',
                                                     'Hungarian Accent',
                                                     'Dutch Accent',
                                                     'French Accent',
                                                     'Polish Accent',
                                                     'Portuguese Accent',
                                                     'Italian Accent'], type="value", default='English Accent', label="Select the Accent of the Speaker"),
                                 gr.inputs.Slider(minimum=-50.0, maximum=50.0, step=0.1, default=0.0, label="DON'T KNOW WHAT IT DOES YET"),
                                 gr.inputs.Slider(minimum=-50.0, maximum=50.0, step=0.1, default=0.0, label="DON'T KNOW WHAT IT DOES YET"),
                                 gr.inputs.Slider(minimum=-50.0, maximum=50.0, step=0.1, default=0.0, label="DON'T KNOW WHAT IT DOES YET"),
                                 gr.inputs.Slider(minimum=-50.0, maximum=50.0, step=0.1, default=0.0, label="DON'T KNOW WHAT IT DOES YET"),
                                 gr.inputs.Slider(minimum=-50.0, maximum=50.0, step=0.1, default=0.0, label="DON'T KNOW WHAT IT DOES YET"),
                                 gr.inputs.Slider(minimum=-50.0, maximum=50.0, step=0.1, default=0.0, label="DON'T KNOW WHAT IT DOES YET")],
                         outputs=gr.outputs.Audio(type="numpy", label=None),
                         layout="vertical",
                         title="Controllable Embeddings",
                         theme="default",
                         allow_flagging="never",
                         allow_screenshot=False,
                         article="")
    iface.launch(enable_queue=True)
