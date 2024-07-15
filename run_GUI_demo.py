import gradio as gr
import torch.cuda

from InferenceInterfaces.ControllableInterface import ControllableInterface
from Utility.utils import float2pcm
from Utility.utils import load_json_from_path


class TTSWebUI:

    def __init__(self, gpu_id="cpu", title="Controllable Text-to-Speech for over 7000 Languages", article="", available_artificial_voices=1000, path_to_iso_list="Preprocessing/multilinguality/iso_to_fullname.json"):
        iso_to_name = load_json_from_path(path_to_iso_list)
        text_selection = [f"{iso_to_name[iso_code]} ({iso_code})" for iso_code in iso_to_name]
        # accent_selection = [f"{iso_to_name[iso_code]} Accent ({iso_code})" for iso_code in iso_to_name]

        self.controllable_ui = ControllableInterface(gpu_id=gpu_id,
                                                     available_artificial_voices=available_artificial_voices)
        self.iface = gr.Interface(fn=self.read,
                                  inputs=[gr.Textbox(lines=2,
                                                     placeholder="write what you want the synthesis to read here...",
                                                     value="What I cannot create, I do not understand.",
                                                     label="Text input"),
                                          gr.Dropdown(text_selection,
                                                      type="value",
                                                      value='English (eng)',
                                                      label="Select the Language of the Text (type on your keyboard to find it quickly)"),
                                          gr.Audio(type="filepath", show_label=True, container=True, label="Voice to Clone (if left empty, will use an artificial voice instead)"),
                                          gr.Slider(minimum=0, maximum=available_artificial_voices, step=1,
                                                    value=279,
                                                    label="Random Seed for the artificial Voice"),
                                          gr.Slider(minimum=0.0, maximum=0.8, step=0.1, value=0.4, label="Prosody Creativity"),
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
             reference_audio,
             voice_seed,
             prosody_creativity,
             duration_scaling_factor,
             pitch_variance_scale,
             energy_variance_scale,
             emb1,
             emb2
             ):
        sr, wav, fig = self.controllable_ui.read(prompt,
                                                 reference_audio,
                                                 language.split(" ")[-1].split("(")[1].split(")")[0],
                                                 language.split(" ")[-1].split("(")[1].split(")")[0],
                                                 voice_seed,
                                                 prosody_creativity,
                                                 duration_scaling_factor,
                                                 1.,
                                                 pitch_variance_scale,
                                                 energy_variance_scale,
                                                 emb1,
                                                 emb2,
                                                 0.,
                                                 0.,
                                                 0.,
                                                 0.,
                                                 -24.)
        return (sr, float2pcm(wav)), fig


if __name__ == '__main__':
    TTSWebUI(gpu_id="cuda" if torch.cuda.is_available() else "cpu")
