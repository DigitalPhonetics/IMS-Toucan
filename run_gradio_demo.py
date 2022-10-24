import random

import gradio as gr
import numpy as np

from InferenceInterfaces.ControllableInterface import ControllableInterface


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


class TTSWebUI:

    def __init__(self, gpu_id="cpu", title="Controllable Embeddings", article=""):
        self.controllable_ui = ControllableInterface(gpu_id=gpu_id)
        self.iface = gr.Interface(fn=self.read,
                                  inputs=[gr.inputs.Textbox(lines=2,
                                                            placeholder="write what you want the synthesis to read here...",
                                                            default="With every sun that sets, I am feeling more, like a stranger on a foreign shore.",
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
                                                              'Italian Text',
                                                              'Chinese Text',
                                                              'Vietnamese Text'], type="value", default='English Text',
                                                             label="Select the Language of the Text"),
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
                                                              'Italian Accent',
                                                              'Chinese Accent',
                                                              'Vietnamese Accent'], type="value",
                                                             default='English Accent',
                                                             label="Select the Accent of the Speaker"),
                                          gr.inputs.Slider(minimum=0, maximum=1000, step=1,
                                                           default=random.randint(0, 1000),
                                                           label="Random Seed for the artificial Voice"),
                                          gr.inputs.Slider(minimum=0.5, maximum=1.5, step=0.1, default=1.0,
                                                           label="Duration Scale"),
                                          gr.inputs.Slider(minimum=0.0, maximum=2.0, step=0.1, default=1.0,
                                                           label="Pause Duration Scale"),
                                          gr.inputs.Slider(minimum=0.0, maximum=2.0, step=0.1, default=1.0,
                                                           label="Pitch Variance Scale"),
                                          gr.inputs.Slider(minimum=0.0, maximum=2.0, step=0.1, default=1.0,
                                                           label="Energy Variance Scale"),
                                          gr.inputs.Slider(minimum=-50.0, maximum=50.0, step=0.1, default=0.0,
                                                           label="Femininity / Masculinity"),
                                          gr.inputs.Slider(minimum=-30.0, maximum=30.0, step=0.1, default=0.0,
                                                           label="Arousal"),
                                          gr.inputs.Slider(minimum=-10.0, maximum=10.0, step=0.1, default=0.0,
                                                           label="Age")
                                          ],
                                  outputs=gr.outputs.Audio(type="numpy", label=None),
                                  layout="vertical",
                                  title=title,
                                  theme="default",
                                  allow_flagging="never",
                                  allow_screenshot=False,
                                  article=article)
        self.iface.launch(enable_queue=True)

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
             emb_slider_6):
        sr, wav = self.controllable_ui.read(prompt,
                                            language,
                                            accent,
                                            voice_seed,
                                            duration_scaling_factor,
                                            pause_duration_scaling_factor,
                                            pitch_variance_scale,
                                            energy_variance_scale,
                                            emb_slider_1,
                                            emb_slider_2,
                                            emb_slider_6)
        return sr, float2pcm(wav.cpu().numpy())


if __name__ == '__main__':
    TTSWebUI(gpu_id="cpu")
