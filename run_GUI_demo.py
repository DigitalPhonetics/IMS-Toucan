import gradio as gr
import torch.cuda

from InferenceInterfaces.ControllableInterface import ControllableInterface
from Preprocessing.multilinguality.SimilaritySolver import load_json_from_path
from Utility.utils import float2pcm


class TTSWebUI:

    def __init__(self, gpu_id="cpu", title="Controllable Text-to-Speech for over 7000 Languages", article="", available_artificial_voices=1000, path_to_iso_list="Preprocessing/multilinguality/iso_to_fullname.json"):
        iso_to_name = load_json_from_path(path_to_iso_list)
        text_selection = [f"{iso_to_name[iso_code]} Text ({iso_code})" for iso_code in iso_to_name]
        # accent_selection = [f"{iso_to_name[iso_code]} Accent ({iso_code})" for iso_code in iso_to_name]

        self.controllable_ui = ControllableInterface(gpu_id=gpu_id,
                                                     available_artificial_voices=available_artificial_voices)
        self.iface = gr.Interface(fn=self.read,
                                  inputs=[gr.Textbox(lines=2,
                                                     placeholder="write what you want the synthesis to read here...",
                                                     value="The woods are lovely, dark and deep, but I have promises to keep, and miles to go, before I sleep.",
                                                     label="Text input"),
                                          gr.Dropdown(text_selection,
                                                      type="value",
                                                      value='English Text (eng)',
                                                      label="Select the Language of the Text (type on your keyboard to find it quickly)"),
                                          gr.Slider(minimum=0, maximum=available_artificial_voices, step=1,
                                                    value=279,
                                                    label="Random Seed for the artificial Voice"),
                                          gr.Slider(minimum=0.7, maximum=1.3, step=0.1, value=1.0, label="Duration Scale"),
                                          gr.Slider(minimum=0.5, maximum=1.5, step=0.1, value=1.0, label="Pitch Variance Scale"),
                                          gr.Slider(minimum=0.5, maximum=1.5, step=0.1, value=1.0, label="Energy Variance Scale"),
                                          gr.Slider(minimum=-10.0, maximum=10.0, step=0.1, value=0.0, label="Femininity / Masculinity"),
                                          gr.Slider(minimum=-10.0, maximum=10.0, step=0.1, value=0.0, label="Voice Depth")
                                          ],
                                  outputs=[gr.Audio(type="numpy", label="Speech"),
                                           gr.Image(label="Visualization")],
                                  title=title,
                                  theme="default",
                                  allow_flagging="never",
                                  article=article)
        self.iface.launch()

    def read(self,
             prompt,
             language,
             voice_seed,
             duration_scaling_factor,
             pitch_variance_scale,
             energy_variance_scale,
             emb1,
             emb2
             ):
        sr, wav, fig = self.controllable_ui.read(
            prompt=prompt,
            language=language.split(" ")[-1].split("(")[1].split(")")[0],
            accent=language.split(" ")[-1].split("(")[1].split(")")[0],
            voice_seed=voice_seed,
            duration_scaling_factor=duration_scaling_factor,
            pause_duration_scaling_factor=1.0,
            pitch_variance_scale=pitch_variance_scale,
            energy_variance_scale=energy_variance_scale,
            emb_slider_1=emb1,
            emb_slider_2=emb2,
            emb_slider_3=0.0,
            emb_slider_4=0.0,
            emb_slider_5=0.0,
            emb_slider_6=0.0
        )
        return (sr, float2pcm(wav)), fig


if __name__ == '__main__':
    TTSWebUI(gpu_id="cuda" if torch.cuda.is_available() else "cpu")
