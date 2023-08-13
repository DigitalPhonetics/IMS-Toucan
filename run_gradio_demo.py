import gradio as gr
import numpy as np

from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface
from Preprocessing.sentence_embeddings.EmotionRoBERTaSentenceEmbeddingExtractor import EmotionRoBERTaSentenceEmbeddingExtractor as SentenceEmbeddingExtractor


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

    def __init__(self, gpu_id="cpu", title="Prompting ToucanTTS", article=""):
        sent_emb_extractor = SentenceEmbeddingExtractor(pooling="cls")
        self.speaker_to_id = {'Female 1': 29,
                                'Female 2': 30,
                                'Female 3': 31,
                                'Female 4': 32,
                                'Male 1': 25,
                                'Male 1': 26,
                                'Male 1': 27,
                                'Male 1': 28,
                                'Male 1': 33,
                                'Male 1': 34}
        self.tts_interface = ToucanTTSInterface(device=gpu_id,
                                                tts_model_path='Proposed',
                                                faster_vocoder=True,
                                                sent_emb_extractor=sent_emb_extractor)
        self.iface = gr.Interface(fn=self.read,
                                  inputs=[gr.Textbox(lines=2,
                                                     placeholder="write what you want the synthesis to read here...",
                                                     value="Today is a beautiful day.",
                                                     label="Text input"),
                                            gr.Textbox(lines=2,
                                                    placeholder="write a (emotional) prompt in order to control the speaking style...",
                                                    value="I am so angry!",
                                                    label="Prompt"),
                                          gr.Dropdown(['Female 1',
                                                       'Female 2',
                                                       'Female 3',
                                                       'Female 4',
                                                       'Male 1',
                                                       'Male 2',
                                                       'Male 3',
                                                       'Male 4',
                                                       'Male 5',
                                                       'Male 6'], type="value",
                                                      value='Female 1', label="Select a Speaker")],
                                  outputs=[gr.Audio(type="numpy", label="Speech"),
                                           gr.Image(label="Visualization")],
                                  title=title,
                                  theme="default",
                                  allow_flagging="never",
                                  article=article)
        self.iface.launch(enable_queue=True)

    def read(self, input, prompt, speaker):
        self.tts_interface.set_language("en")
        self.tts_interface.set_speaker_id(self.speaker_to_id[speaker])
        self.tts_interface.set_sentence_embedding(prompt)
        wav, fig = self.tts_interface(input, return_plot_as_filepath=True)
        return (24000, float2pcm(wav.cpu().numpy())), fig

if __name__ == '__main__':
    TTSWebUI(gpu_id="cpu")