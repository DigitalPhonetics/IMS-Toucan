import os

from Utility.EvaluationScripts.SpeakerVisualization import Visualizer
from run_text_to_file_reader import read_texts_as_ensemble


def visualize_random_speakers(generate=False):
    if generate:
        # first we generate a bunch of audios with random speaker embeddings
        if not len(os.listdir("audios/random_speakers/")) != 0:
            os.makedirs("audios/random_speakers/", exist_ok=True)
            read_texts_as_ensemble(model_id="Libri", sentence="Hi, I am a completely random speaker that probably doesn't exist!",
                                   filename="audios/random_speakers/libri", amount=100)

    # then we visualize those audios
    vs = Visualizer()
    ltf = dict()
    for audio_file in os.listdir("audios/random_speakers"):
        if audio_file not in ltf:
            ltf[audio_file] = list()
        ltf[audio_file].append(f"audios/random_speakers/{audio_file}")
    vs.visualize_speaker_embeddings(label_to_filepaths=ltf, title_of_plot="Embeddings of TTS with random Condition")


def visualize_libritts():
    vs = Visualizer()
    ltf = dict()
    for speaker in os.listdir("audios/LibriTTS"):
        ltf[speaker] = list()
        for book in os.listdir(f"audios/LibriTTS/{speaker}"):
            for audio_file in os.listdir(f"audios/LibriTTS/{speaker}/{book}"):
                ltf[speaker].append(f"audios/LibriTTS/{speaker}/{book}/{audio_file}")
    vs.visualize_speaker_embeddings(label_to_filepaths=ltf, title_of_plot="Embeddings of a Subset of LibriTTS")


if __name__ == '__main__':
    visualize_random_speakers()
