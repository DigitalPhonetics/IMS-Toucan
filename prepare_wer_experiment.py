import os

import torch

from InferenceInterfaces.Eva_FastSpeech2 import Eva_FastSpeech2
from InferenceInterfaces.Eva_Tacotron2 import Eva_Tacotron2
from InferenceInterfaces.HokusPokus_FastSpeech2 import HokusPokus_FastSpeech2
from InferenceInterfaces.HokusPokus_Tacotron2 import HokusPokus_Tacotron2
from InferenceInterfaces.Karlsson_FastSpeech2 import Karlsson_FastSpeech2
from InferenceInterfaces.Karlsson_Tacotron2 import Karlsson_Tacotron2
from InferenceInterfaces.LowRes_FastSpeech2 import LowRes_FastSpeech2 as fast_low
from InferenceInterfaces.LowRes_Tacotron2 import LowRes_Tacotron2 as taco_low
from InferenceInterfaces.Nancy_FastSpeech2 import Nancy_FastSpeech2
from InferenceInterfaces.Nancy_Tacotron2 import Nancy_Tacotron2

tts_dict = {
    "fast_nancy"   : Nancy_FastSpeech2,
    "fast_hokus"   : HokusPokus_FastSpeech2,

    "taco_nancy"   : Nancy_Tacotron2,
    "taco_hokus"   : HokusPokus_Tacotron2,

    "taco_low"     : taco_low,
    "fast_low"     : fast_low,

    "taco_eva"     : Eva_Tacotron2,
    "fast_eva"     : Eva_FastSpeech2,

    "taco_karlsson": Karlsson_Tacotron2,
    "fast_karlsson": Karlsson_FastSpeech2,
    }


def read_texts(model_id, sentence, filename, device="cpu"):
    tts = tts_dict[model_id](device=device)
    if type(sentence) == str:
        sentence = [sentence]
    tts.read_to_file(text_list=sentence, file_location=filename)
    del tts


def save_weights(model_id):
    tts_dict[model_id](device="cpu").save_pretrained_weights()


def read_harvard_sentences(model_id, device):
    tts = tts_dict[model_id](device=device)

    with open("Utility/test_sentences_combined_3.txt", "r", encoding="utf8") as f:
        sents = f.read().split("\n")
    output_dir = "audios/harvard_03_{}".format(model_id)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for index, sent in enumerate(sents):
        tts.read_to_file(text_list=[sent], file_location=output_dir + "/{}.wav".format(index))

    with open("Utility/test_sentences_combined_6.txt", "r", encoding="utf8") as f:
        sents = f.read().split("\n")
    output_dir = "audios/harvard_06_{}".format(model_id)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for index, sent in enumerate(sents):
        tts.read_to_file(text_list=[sent], file_location=output_dir + "/{}.wav".format(index))


if __name__ == '__main__':

    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("audios", exist_ok=True)

    texts = """Der endgültige Status der palästinensischen Autonomie-Gebiete stehe nicht im Mittelpunkt der Unterredungen.
In der Diskussion über den Verfassungsprozess der Europäischen Union setzt Bundeskanzlerin Merkel weiter auf eine einvernehmliche Verständigung der Mitgliedsstaaten.
Die Behörden gaben eine Tsunami-Warnung für die Präfektur Ishikawa an der Westküste Japans heraus.
Auch Großbritannien beharrt auf dem festgelegten Termin und will andernfalls die Direktverwaltung Nordirlands aus London fortsetzen.
Im Tagesverlauf Aufheiterungen und meist trocken.
Bei den Tarifverhandlungen für die Metall- und Elektroindustrie haben die Arbeitgeber in Baden-Württemberg eine Lohnerhöhung von 2,5 Prozent angeboten.
Dies würde einen echten Fortschritt für die Kunden bedeuten.
Morgen im Norden und in der Mitte heiter oder sonnig nach Süden hin anfangs wolkig oder stark bewölkt 8 bis 16 Grad.
Am Nachmittag soll über die für Mitte Juni geplante Verschmelzung beider Parteien abgestimmt werden.
Der argentinische Präsident Kirchner hat der Justiz seines Landes vorgeworfen, die Strafverfolgung von Menschenrechtsverletzungen aus der Zeit der Militärdiktatur zu sabotieren.
Auch heute sei die Sklaverei in vielen Teilen der Welt noch gängige Praxis.
Das Verfahren vor einem Militärtribunal gegen den australischen Guantanamo-Häftling Hicks ist vertagt worden.
Die Nacht war damit eine Stunde kürzer.
Er finde es verbrecherisch, dass der Regierende Bürgermeister Wowereit sich derart mit den Erben der DDR-Nomenklatura eingelassen habe, sagte Biermann in seiner Dankesrede im Roten Rathaus.
Einen Tag nach dem Sondergipfel der Europäischen Union in Berlin hält die Diskussion um den Verfassungsprozess an.
Die 27 EU-Staats- und Regierungschefs hatten bis zuletzt um die Formulierungen gerungen.
Vor allem Tschechien und Polen hatten sich nach dem EU-Sondergipfel in Berlin kritisch über das Beratungsverfahren geäußert.
Bundeswirtschaftsminister Glos lehnte eine entsprechende Regelung ab.
Vorgesehen ist, dass grenzüberschreitende Überweisungen künftig so kostengünstig und sicher sein sollen, wie innerhalb eines Mitgliedstaates.
Das Bundeskartellamt muss das Geschäft noch billigen.
Durch die geplanten Verfassungsänderungen erhält die Polizei weit reichende Befugnisse bei Festnahmen und Überwachungen.
Als Reaktion auf die vom UNO-Sicherheitsrat beschlossene Verschärfung der Sanktionen hat der Iran die Zusammenarbeit mit der Internationalen Atomenergieorganisation eingeschränkt.
Im Übernahmekampf um den Energieversorger Endesa dürfen die beiden Großaktionäre Enel und Acciona in den nächsten sechs Monaten keine Gegenofferte zum E.ON-Gebot abgeben.
Der Angeklagte hatte zwischen 2004 und 5 rund 400 Tonnen sogenanntes Gammelfleisch in Umlauf gebracht, das für den Verzehr nicht mehr geeignet und irreführend ausgezeichnet war.
Er teile Münteferings Auffassung, wonach dies gegeben sei, wenn Gehälter um 30 Prozent unter dem üblichen Tarif lägen.
In Ägypten hat die Regierung die Sicherheitsvorkehrungen verstärkt, um Proteste gegen das heute stattfindende Verfassungsreferendum zu verhindern.
Und hier noch einmal die Übersicht.
Die internationale Gemeinschaft solle aber zunächst militärisch und zivil im Kosovo präsent bleiben und so die Unabhängigkeit überwachen.
Deutschland sollte nach Ansicht von Bundesfinanzminister Steinbrück die Sparanstrengungen anderer Länder der Euro-Zone zum Vorbild nehmen.
Die in Fahrzeugen versteckten Sprengsätze detonierten etwa gleichzeitig.
Der luxemburgische Premierminister Juncker hat die Eu kurz vor Unterzeichnung der Berliner Erklärung als Glücksfall für Europa und die Welt bezeichnet.
Schwacher bis mäßiger Ostwind, an der See und im Bergland starke Böen.
In Deutschland gilt wieder die Sommerzeit.
Man brauche offene Postmärkte, sagte Glos in München und fügte hinzu, dies sei ein Vorteil für die Verbraucher.
Der italienische Senat hat am Abend in Rom das weitere Engagement des Landes in Afghanistan gebilligt.
Dieser dürfe nicht immer wieder durch diverse Regierungswechsel zur Disposition gestellt werden.
Zugleich wandte er sich gegen Forderungen nach Entschädigung, da auch Afrikaner im Sklavenhandel tätig gewesen seien.
Die Vereinigten Staaten und Israel sind sich offenbar uneins über den weiteren Friedensprozess im Nahen Osten.
Milan Baros erzielte den Anschlusstreffer.
Bei der Verleihung der Ehrenbürgerwürde der Stadt Berlin an Wolf Biermann will der Regierende Bürgermeister Wowereit auf die Kritik des Liedermachers eingehen.
Baden-Württemberg gilt als möglicher Pilotbezirk.
Eine angekündigte Pressekonferenz von Außenministerin Rice nach einem Meinungsaustausch mit dem israelischen Regierungschef Olmert wurde abgesagt.
Bei einem Festakt in Berlin zum fünfzigsten Jahrestag der Römischen Verträge unterzeichneten EU-Ratspräsidentin Merkel, Kommissionspräsident Barroso und der Präsident des Europäischen Parlaments, Pöttering, stellvertretend für die Staats- und Regierungschefs die " Berliner Erklärung ".
Die Karlsruher Richter hatten entschieden, dass heimliche Vaterschaftstests auch künftig nicht als Beweismittel anerkannt werden.
Wie das amerikanische Militär mitteilte, gelten sie als Anführer einer Terrorgruppe, die mit Sprengstoffanschlägen seit November rund 900 irakische Zivilisten getötet haben soll.
Sie widerspreche dem Rechtsempfinden von mindestens 80 Prozent der Menschen in Deutschland.
Im Mittelpunkt steht eine von der Ratspräsidentin, Bundeskanzlerin Merkel, vorbereitete " Berliner Erklärung ".
Dabei wurden nach einer neuen Bilanz mehr als 190 Menschen verletzt, eine Frau starb.
Diese Einschränkung verstößt nach Ansicht des saarländischen Finanzgerichts gegen das Gleichheitsgebot des Grundgesetzes.
Die EU-Kommission hatte Anfang März neun kleineren russischen Fluggesellschaften wegen Sicherheitsmängeln die Landeerlaubnis in allen EU-Ländern entzogen.
Der Staat müsse sich vor allem um die Schaffung neuer Arbeitsplätze kümmern und dürfe nicht die Lohnfindung an sich ziehen.""".split("\n")

    tts1 = tts_dict["taco_karlsson"](device=exec_device)
    tts2 = tts_dict["fast_karlsson"](device=exec_device)
    tts3 = tts_dict["taco_low"](device=exec_device)
    tts4 = tts_dict["fast_low"](device=exec_device)

    for index, text in enumerate(texts):

        if text.strip() != "":
            tts1.read_to_file(text_list=[text], file_location=f"audios/1_{index}.wav")
            tts2.read_to_file(text_list=[text], file_location=f"audios/2_{index}.wav")
            tts3.read_to_file(text_list=[text], file_location=f"audios/3_{index}.wav")
            tts4.read_to_file(text_list=[text], file_location=f"audios/4_{index}.wav")
