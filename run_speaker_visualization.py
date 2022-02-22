import os

from Utility.EvaluationScripts.SpeakerVisualization import Visualizer
from run_text_to_file_reader import read_texts_as_ensemble

if __name__ == '__main__':

    # first we generate a bunch of audios with random speaker embeddings
    if not len(os.listdir("audios/random_speakers/")) != 0:
        os.makedirs("audios/random_speakers/", exist_ok=True)
        read_texts_as_ensemble(model_id="Libri", sentence="Hi, I am a completely random speaker that probably doesn't exist!",
                               filename="audios/random_speakers/libri", amount=100)

    # then we visualize those audios
    vs = Visualizer()
    ltf = dict()
    for audio_file in os.listdir("audios/random_speakers"):
        ltf[audio_file] = f"audios/random_speakers/{audio_file}"
    vs.visualize_speaker_embeddings(label_to_filepaths=ltf, title_of_plot="Embeddings of TTS with random Condition")
