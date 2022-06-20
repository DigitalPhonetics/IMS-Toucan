import os

import soundfile as sf

from Preprocessing.GSTExtractor import ProsodicConditionExtractor
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


def visualize_libritts(model_id):
    vs = Visualizer(model_id=model_id)
    ltf = dict()
    for speaker in os.listdir("audios/LibriTTS"):
        ltf[speaker] = list()
        for book in os.listdir(f"audios/LibriTTS/{speaker}"):
            for audio_file in os.listdir(f"audios/LibriTTS/{speaker}/{book}"):
                ltf[speaker].append(f"audios/LibriTTS/{speaker}/{book}/{audio_file}")
    vs.visualize_speaker_embeddings(label_to_filepaths=ltf, title_of_plot=f"Some LibriTTS Embeddings - {model_id}")


def visualize_adept_experiment():
    vs = Visualizer()
    ltf = dict()
    for exp in os.listdir("audios/adept_plot"):
        for sample in os.listdir(f"audios/adept_plot/{exp}"):

            spk_id = sample.split("_")[1].split(".")[0]
            if spk_id == "ad00":
                spk_label = "Female"
            elif spk_id == "ad01":
                spk_label = "Male"
            else:
                spk_label = "Other Female"

            if exp == "human":
                exp_label = "Human"
            elif exp == "same_voice_diff_style":
                exp_label = "Unconditioned"
            else:
                exp_label = "Cloned"

            plot_label = f"{spk_label} - {exp_label}"

            if exp_label != "Human" and spk_label != "Other Female":
                if plot_label not in ltf:
                    ltf[plot_label] = list()
                ltf[plot_label].append(f"audios/adept_plot/{exp}/{sample}")

    vs.visualize_speaker_embeddings(label_to_filepaths=ltf,
                                    title_of_plot="Speakers with and without Cloning",
                                    include_pca=False,
                                    colors=["limegreen", "darkgreen", "dodgerblue", "darkblue"])


def visualize_speakers_languages_crossover():
    ltf = dict()
    vs = Visualizer()
    for file in os.listdir("audios/speakers_for_plotting"):
        label = file.split("_")[0].capitalize() + " Speaker"
        if label not in ltf:
            ltf[label] = list()
        ltf[label].append(f"audios/speakers_for_plotting/{file}")
    vs.visualize_speaker_embeddings(label_to_filepaths=ltf, title_of_plot="Speakers Across Languages", include_pca=False)


def calculate_spk_sims_multiling():
    ltf = dict()
    vs = Visualizer()
    for file in os.listdir("audios/speakers_for_plotting"):
        label = file.split("_")[0]
        if label not in ltf:
            ltf[label] = list()
        ltf[label].append(f"audios/speakers_for_plotting/{file}")
    for reference in os.listdir("audios/multilanguage_references"):
        label = reference.split(".")[0]
        print(label)
        print(vs.calculate_spk_sim(f"audios/multilanguage_references/{reference}", ltf[label]))


def _test_speaker_embedding_extraction():
    wave, sr = sf.read("audios/speaker_references_for_testing/female_mid_voice.wav")
    ext = ProsodicConditionExtractor(sr=sr, model_id="LibriGST")
    print(ext.extract_condition_from_reference_wave(wave=wave).shape)


if __name__ == '__main__':
    visualize_libritts(model_id="LibriGST")
    visualize_libritts(model_id="LibriGST_no_cycle")
