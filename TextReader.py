import torch

from InferenceInterfaces.SingleSpeakerTransformerTTSInference import SingleSpeakerTransformerTTSInference


def read_texts(lang, sentence, filename, reduction_factor=1, device="cpu"):
    tts = SingleSpeakerTransformerTTSInference(lang=lang, reduction_factor=reduction_factor, device=device)
    if type(sentence) == str:
        sentence = [sentence]
    tts.read_to_file(text_list=sentence, file_location=filename)
    del tts


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    read_texts(lang="de",
               sentence=[
                   "Ein Vater hatte zwei Söhne, davon war der älteste klug und gescheit, und wußte sich in alles wohl zu schicken.",
                   "Der jüngste aber war dumm, konnte nichts begreifen und lernen, und wenn ihn die Leute sahen, sprachen sie:",
                   "Mit dem wird der Vater noch seine Last haben!",
                   "Wenn nun etwas zu tun war, so mußte es der älteste allzeit ausrichten;",
                   "Hieß ihn aber der Vater noch spät oder gar in der Nacht etwas holen, und der Weg ging dabei über den Kirchhof oder sonst einen schaurigen Ort, so antwortete er wohl:",
                   "Ach nein, Vater, ich gehe nicht dahin, es gruselt mir!", "Denn er fürchtete sich.",
                   "Oder wenn abends beim Feuer Geschichten erzählt wurden, wobei einem die Haut schaudert, so sprachen die Zuhörer manchmal:",
                   "Ach, es gruselt mir!",
                   "Der jüngste saß in einer Ecke und hörte das mit an und konnte nicht begreifen, was es heißen sollte.",
                   "Immer sagen sie, es gruselt mir, es gruselt mir! Mir gruselt's nicht.",
                   "Das wird wohl eine Kunst sein, von der ich auch nichts verstehe."],
               filename="test_de.wav",
               reduction_factor=1,
               device=device)

    read_texts(lang="en",
               sentence=[
                   "Lying in a field of glass, underneath the overpass.",
                   "Mangled in the shards of a metal frame.",
                   "Woken from the dream by my own name."],
               filename="test_en.wav",
               reduction_factor=1,
               device=device)
