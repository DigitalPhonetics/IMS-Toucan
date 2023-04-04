import os

import gradio as gr
import numpy
import numpy as np
import pyloudnorm as pyln
import torch
from pedalboard import HighpassFilter
from pedalboard import LowpassFilter
from pedalboard import Pedalboard

from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface


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


class Demo:

    def __init__(self, gpu_id="cpu", title="Blizzard Challenge 2023 - Team IMS", article=""):
        self.controllable_ui = BlizzardInterface(gpu_id=gpu_id)
        self.iface = gr.Interface(fn=self.controllable_ui.read,
                                  inputs=[gr.Textbox(lines=2,
                                                     placeholder="write what you want the synthesis to read here...",
                                                     value="Ces cerises sont si sûres qu'on ne sait pas si c'en sont. ",
                                                     label="Text input"),
                                          gr.Dropdown(['Nadine Eckert-Boulet',
                                                       'Aurélie Derbier',
                                                       ], type="value", value='Nadine Eckert-Boulet', label="Select the Speaker of the Text"),
                                          gr.Slider(minimum=0.5, maximum=1.5, step=0.1, value=1.0, label="Duration Scale"),
                                          gr.Slider(minimum=0.0, maximum=2.0, step=0.1, value=1.0, label="Pause Duration Scale"),
                                          gr.Slider(minimum=0.0, maximum=2.0, step=0.1, value=1.0, label="Pitch Variance Scale"),
                                          gr.Slider(minimum=0.0, maximum=2.0, step=0.1, value=1.0, label="Energy Variance Scale")
                                          ],
                                  outputs=[gr.Audio(type="numpy", label="Speech")],
                                  title=title,
                                  theme="default",
                                  allow_flagging="never",
                                  article=article)
        self.iface.launch(enable_queue=True)


class BlizzardInterface:

    def __init__(self, gpu_id="cpu"):
        if gpu_id == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        else:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ad_model = ToucanTTSInterface(device=self.device, tts_model_path="AD", faster_vocoder=True, language="fr")
        self.neb_model = ToucanTTSInterface(device=self.device, tts_model_path="NEB", faster_vocoder=True, language="fr")
        self.ad_model.set_utterance_embedding(path_to_reference_audio="audios/blizzard_references/AD_REFERENCE.wav")
        self.neb_model.set_utterance_embedding(path_to_reference_audio="audios/blizzard_references/NEB_REFERENCE.wav")
        self.ad_effects = Pedalboard(plugins=[HighpassFilter(cutoff_frequency_hz=200),
                                              LowpassFilter(cutoff_frequency_hz=12000)])
        self.neb_effects = Pedalboard(plugins=[HighpassFilter(cutoff_frequency_hz=60),
                                               LowpassFilter(cutoff_frequency_hz=12000)])
        self.meter = pyln.Meter(48000)

    def read(self,
             prompt,
             ad_neb,
             duration_scaling_factor,
             pause_duration_scaling_factor,
             pitch_variance_scale,
             energy_variance_scale
             ):

        if ad_neb == "Aurélie Derbier":
            model = self.ad_model
        else:
            model = self.neb_model

        prompt = prompt.replace("§", " ").replace("#", " ").replace("»", '"').replace("«", '"').replace(":", ",").replace("  ", " ")
        silence = torch.zeros([6000])

        phones = model.text2phone.get_phone_string(prompt)
        if len(phones) > 1800:
            prompt = "Votre saisie était trop longue. Veuillez essayer un texte plus court ou le diviser en plusieurs parties."
        wav = model(prompt,
                    input_is_phones=False,
                    duration_scaling_factor=duration_scaling_factor,
                    pitch_variance_scale=pitch_variance_scale,
                    energy_variance_scale=energy_variance_scale,
                    pause_duration_scaling_factor=pause_duration_scaling_factor,
                    return_plot_as_filepath=False).cpu()
        wav = torch.cat((silence, wav, silence), 0).cpu().numpy()
        wav = numpy.array([val for val in wav for _ in (0, 1)])
        sr = 48000
        if ad_neb == "Aurélie Derbier":
            wav = self.ad_effects(wav, sr)
        else:
            wav = self.neb_effects(wav, sr)
        loudness = self.meter.integrated_loudness(wav)
        if ad_neb == "Aurélie Derbier":
            wav = pyln.normalize.loudness(wav, loudness, -33.0)
        else:
            wav = pyln.normalize.loudness(wav, loudness, -29.0)
        wav = float2pcm(wav)

        return 48000, wav


if __name__ == "__main__":
    Demo()
