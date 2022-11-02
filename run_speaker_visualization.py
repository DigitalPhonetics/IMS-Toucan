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
            read_texts_as_ensemble(model_id="Meta",
                                   sentence="Hi, I am a completely random speaker that probably doesn't exist!",
                                   filename="audios/random_speakers/libri", amount=100)

    # then we visualize those audios
    vs = Visualizer()
    ltf = dict()
    for audio_file in os.listdir("audios/random_speakers"):
        if audio_file not in ltf:
            ltf[audio_file] = list()
        ltf[audio_file].append(f"audios/random_speakers/{audio_file}")
    vs.visualize_speaker_embeddings(label_to_filepaths=ltf, title_of_plot="Embeddings of TTS with random Condition")


def visualize_libritts(save_file_path):
    vs = Visualizer()
    ltf = dict()
    for speaker in os.listdir("audios/LibriTTS"):
        if os.path.isdir(f"audios/LibriTTS/{speaker}"):
            ltf[speaker] = list()
            for book in os.listdir(f"audios/LibriTTS/{speaker}"):
                for audio_file in os.listdir(f"audios/LibriTTS/{speaker}/{book}"):
                    if audio_file.endswith(".wav"):
                        ltf[speaker].append(f"audios/LibriTTS/{speaker}/{book}/{audio_file}")
                    if len(ltf[speaker]) > 5:
                        break
    vs.visualize_speaker_embeddings(label_to_filepaths=ltf, title_of_plot=f"Some LibriTTS Embeddings",
                                    save_file_path=save_file_path)


def visualize_ravdess():
    vs = Visualizer()
    ltf = dict()
    for speaker in os.listdir("audios/RAVDESS_male"):
        ltf[speaker] = list()
        for audio_file in os.listdir(f"audios/RAVDESS_male/{speaker}"):
            ltf[speaker].append(f"audios/RAVDESS_male/{speaker}/{audio_file}")
    vs.visualize_speaker_embeddings(label_to_filepaths=ltf, title_of_plot=f"RAVDESS Male Embeddings")
    vs = Visualizer()
    ltf = dict()
    for speaker in os.listdir("audios/RAVDESS_female"):
        ltf[speaker] = list()
        for audio_file in os.listdir(f"audios/RAVDESS_female/{speaker}"):
            ltf[speaker].append(f"audios/RAVDESS_female/{speaker}/{audio_file}")
    vs.visualize_speaker_embeddings(label_to_filepaths=ltf, title_of_plot=f"RAVDESS Female Embeddings")


def visualize_speakers_and_emotions():
    # small emotion test
    # vs = Visualizer()
    # ltf = dict()
    # for speaker in os.listdir("audios/emotions"):
    #    ltf[speaker] = list()
    #    for audio_file in os.listdir(f"audios/emotions/{speaker}"):
    #        ltf[speaker].append(f"audios/emotions/{speaker}/{audio_file}")
    # vs.visualize_speaker_embeddings(label_to_filepaths=ltf, title_of_plot=f"Embeddings by Emotion")

    # iemocap
    vs = Visualizer()
    ltf = {"neu": [], "sad": [], "ang": [], "hap": []}
    with open("audios/IEMOCAP/iemocap_full_dataset.csv", "r", encoding="utf8") as f:
        lines = f.read().split("\n")
    for line in lines:
        if line.strip() != "":
            if line.split(",")[3] in ["neu", "sad", "ang", "hap"] and line.split(",")[1] == "script" \
                    and line.split(",")[0] == "1" and line.split(",")[-2] == "3":
                ltf[line.split(",")[3]].append(f"audios/IEMOCAP/{line.split(',')[-1].split('/')[-2]}/{line.split(',')[-1].split('/')[-1]}")
    print(ltf)
    vs.visualize_speaker_embeddings(label_to_filepaths=ltf, title_of_plot=f"Embeddings by Emotion")

    # multiling speakers
    vs = Visualizer()
    ltf = dict()
    for speaker in os.listdir("audios/speakers"):
        ltf[speaker] = list()
        for audio_file in os.listdir(f"audios/speakers/{speaker}"):
            ltf[speaker].append(f"audios/speakers/{speaker}/{audio_file}")
    vs.visualize_speaker_embeddings(label_to_filepaths=ltf, title_of_plot=f"Embeddings by Speaker")


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
    ext = ProsodicConditionExtractor(sr=sr)
    print(ext.extract_condition_from_reference_wave(wave=wave).shape)


def check_same_params(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


if __name__ == '__main__':
    visualize_libritts("libri_speakers_plotted.png")
