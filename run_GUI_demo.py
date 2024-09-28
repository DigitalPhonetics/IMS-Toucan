import gradio as gr
import torch.cuda
from huggingface_hub import hf_hub_download

from InferenceInterfaces.ControllableInterface import ControllableInterface
from Utility.utils import float2pcm
from Utility.utils import load_json_from_path


class TTSWebUI:

    def __init__(self,
                 gpu_id="cpu",
                 title="Controllable Text-to-Speech for over 7000 Languages",
                 article="",
                 tts_model_path=None,
                 vocoder_model_path=None,
                 embedding_gan_path=None,
                 available_artificial_voices=50  # be careful with this, if you want too many, it might lead to an endless loop
                 ):
        path_to_iso_list = hf_hub_download(repo_id="Flux9665/ToucanTTS", filename="iso_to_fullname.json")
        iso_to_name = load_json_from_path(path_to_iso_list)
        text_selection = [f"{iso_to_name[iso_code]} ({iso_code})" for iso_code in iso_to_name]
        # accent_selection = [f"{iso_to_name[iso_code]} Accent ({iso_code})" for iso_code in iso_to_name]
        if tts_model_path is None:
            tts_model_path = hf_hub_download(repo_id="Flux9665/ToucanTTS", filename="ToucanTTS.pt")
        if vocoder_model_path is None:
            vocoder_model_path = hf_hub_download(repo_id="Flux9665/ToucanTTS", filename="Vocoder.pt")
        if embedding_gan_path is None:
            embedding_gan_path = hf_hub_download(repo_id="Flux9665/ToucanTTS", filename="embedding_gan.pt")

        self.controllable_ui = ControllableInterface(gpu_id=gpu_id,
                                                     available_artificial_voices=available_artificial_voices,
                                                     tts_model_path=tts_model_path,
                                                     vocoder_model_path=vocoder_model_path,
                                                     embedding_gan_path=embedding_gan_path)
        self.iface = gr.Interface(fn=self.read,
                                  inputs=[gr.Textbox(lines=2,
                                                     placeholder="write what you want the synthesis to read here...",
                                                     value="What I cannot create, I do not understand.",
                                                     label="Text input"),
                                          gr.Dropdown(text_selection,
                                                      type="value",
                                                      value='English (eng)',
                                                      label="Select the Language of the Text (type on your keyboard to find it quickly)"),
                                          gr.Slider(minimum=0.0, maximum=0.8, step=0.1, value=0.5, label="Prosody Creativity"),
                                          gr.Slider(minimum=0.7, maximum=1.3, step=0.1, value=1.0, label="Faster - Slower"),
                                          gr.Slider(minimum=0, maximum=available_artificial_voices, step=1, value=27, label="Random Seed for the artificial Voice"),
                                          gr.Slider(minimum=-10.0, maximum=10.0, step=0.1, value=0.0, label="Gender of artificial Voice"),
                                          gr.Audio(type="filepath", show_label=True, container=True, label="[OPTIONAL] Voice to Clone (if left empty, will use an artificial voice instead)"),
                                          # gr.Slider(minimum=0.5, maximum=1.5, step=0.1, value=1.0, label="Pitch Variance Scale"),
                                          # gr.Slider(minimum=0.5, maximum=1.5, step=0.1, value=1.0, label="Energy Variance Scale"),
                                          # gr.Slider(minimum=-10.0, maximum=10.0, step=0.1, value=0.0, label="Voice Depth")
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
             prosody_creativity,
             duration_scaling_factor,
             voice_seed,
             emb1,
             reference_audio,
             # pitch_variance_scale,
             # energy_variance_scale,
             # emb2
             ):
        sr, wav, fig = self.controllable_ui.read(prompt,
                                                 reference_audio,
                                                 language.split(" ")[-1].split("(")[1].split(")")[0],
                                                 language.split(" ")[-1].split("(")[1].split(")")[0],
                                                 voice_seed,
                                                 prosody_creativity,
                                                 duration_scaling_factor,
                                                 1.,
                                                 1.0,
                                                 1.0,
                                                 emb1,
                                                 0.,
                                                 0.,
                                                 0.,
                                                 0.,
                                                 0.,
                                                 -12.)
        return (sr, float2pcm(wav)), fig


if __name__ == '__main__':
    TTSWebUI(gpu_id="cuda" if torch.cuda.is_available() else "cpu")
