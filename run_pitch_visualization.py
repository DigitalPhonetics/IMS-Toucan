from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
from Utility.EvaluationScripts.audio_vs_audio import get_pitch_curve_diff_extractors

tf = ArticulatoryCombinedTextFrontend(language='en')
path = "/Users/kockja/Documents/PhD/adept/human/2.wav"
transcript = "Don't forget to shut the door behind you?"
text = tf.string_to_tensor(transcript, handle_missing=False).squeeze(0)

# get_pitch_curves(path_1, path_2, plot_curves=True)
# get_pitch_curves_abc(path_1, path_2, path_3)
get_pitch_curve_diff_extractors(path, text)
