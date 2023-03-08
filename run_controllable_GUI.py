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

    def __init__(self, gpu_id="cpu", title="Controllable Embeddings", article="", available_artificial_voices=1000):
        self.controllable_ui = ControllableInterface(gpu_id=gpu_id,
                                                     available_artificial_voices=available_artificial_voices)
        self.iface = gr.Interface(fn=self.read,
                                  inputs=[gr.Textbox(lines=2,
                                                     placeholder="write what you want the synthesis to read here...",
                                                     value="Colorless green ideas sleep furiously!",
                                                     label="Text input"),
                                          gr.Dropdown(['English Text',
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
                                                       'Vietnamese Text'], type="value", value='English Text', label="Select the Language of the Text"),
                                          gr.Dropdown(['English Accent',
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
                                                      value='English Accent', label="Select the Accent of the Speaker"),
                                          gr.Textbox(lines=3,
                                                     placeholder="\nThe sliders below control the speaker embedding",
                                                     value="\nThe sliders below control the speaker embedding",
                                                     label=" ",
                                                     show_label=False),
                                          gr.Slider(minimum=0, maximum=available_artificial_voices, step=1,
                                                    value=279,
                                                    label="Random Seed for the artificial Voice"),
                                          gr.Slider(minimum=-50.0, maximum=50.0, step=0.1, value=0.0, label="Femininity / Masculinity"),
                                          gr.Slider(minimum=-30.0, maximum=30.0, step=0.1, value=0.0, label="Sibilance"),
                                          gr.Slider(minimum=-30.0, maximum=30.0, step=0.1, value=0.0, label="Accentuated High / Low Frequencies"),
                                          gr.Slider(minimum=-30.0, maximum=30.0, step=0.1, value=0.0, label="Loudness / Arousal / Calmness"),
                                          gr.Slider(minimum=-20.0, maximum=20.0, step=0.1, value=0.0, label="Tone / Timbre"),
                                          gr.Textbox(lines=3,
                                                     placeholder="\nThe sliders below directly control the TTS",
                                                     value="\nThe sliders below directly control the TTS",
                                                     label=" ",
                                                     show_label=False),
                                          gr.Slider(minimum=0.5, maximum=1.5, step=0.1, value=1.0, label="Duration Scale"),
                                          gr.Slider(minimum=0.0, maximum=2.0, step=0.1, value=1.0, label="Pause Duration Scale"),
                                          gr.Slider(minimum=0.0, maximum=2.0, step=0.1, value=1.0, label="Pitch Variance Scale"),
                                          gr.Slider(minimum=0.0, maximum=2.0, step=0.1, value=1.0, label="Energy Variance Scale")
                                          ],
                                  outputs=[gr.Audio(type="numpy", label="Speech"),
                                           gr.Image(label="Visualization")],
                                  title=title,
                                  theme="default",
                                  allow_flagging="never",
                                  article=article)
        self.iface.launch(enable_queue=True)

    def read(self,
             prompt,
             language,
             accent,
             ignore_1,
             voice_seed,
             emb1,
             emb2,
             emb3,
             emb5,
             emb6,
             ignore_2,
             duration_scaling_factor,
             pause_duration_scaling_factor,
             pitch_variance_scale,
             energy_variance_scale):
        sr, wav, fig = self.controllable_ui.read(prompt,
                                                 language,
                                                 accent,
                                                 voice_seed,
                                                 duration_scaling_factor,
                                                 pause_duration_scaling_factor,
                                                 pitch_variance_scale,
                                                 energy_variance_scale,
                                                 emb1,
                                                 emb2,
                                                 emb3,
                                                 0.0,  # slider 4 did not have a meaningful interpretation, too many properties mixed
                                                 emb5,
                                                 emb6)
        return (sr, float2pcm(wav.cpu().numpy())), fig


if __name__ == '__main__':
    TTSWebUI(gpu_id="cpu")
