import os

import torch

from InferenceInterfaces.Elizabeth_FastSpeechInference import Elizabeth_FastSpeechInference
from InferenceInterfaces.Elizabeth_TransformerTTSInference import Elizabeth_TransformerTTSInference
from InferenceInterfaces.Eva_FastSpeechInference import Eva_FastSpeechInference
from InferenceInterfaces.Eva_TransformerTTSInference import Eva_TransformerTTSInference
from InferenceInterfaces.Karlsson_FastSpeechInference import Karlsson_FastSpeechInference
from InferenceInterfaces.Karlsson_TransformerTTSInference import Karlsson_TransformerTTSInference
from InferenceInterfaces.LJSpeech_FastSpeechInference import LJSpeech_FastSpeechInference
from InferenceInterfaces.LJSpeech_TransformerTTSInference import LJSpeech_TransformerTTSInference
from InferenceInterfaces.LibriTTS_FastSpeechInference import LibriTTS_FastSpeechInference
from InferenceInterfaces.LibriTTS_TransformerTTSInference import LibriTTS_TransformerTTSInference
from InferenceInterfaces.Thorsten_FastSpeechInference import Thorsten_FastSpeechInference
from InferenceInterfaces.Thorsten_TransformerTTSInference import Thorsten_TransformerTTSInference

tts_dict = {
    "fast_thorsten"  : Thorsten_FastSpeechInference,
    "fast_lj"        : LJSpeech_FastSpeechInference,
    "fast_libri"     : LibriTTS_FastSpeechInference,
    "fast_karl"      : Karlsson_FastSpeechInference,
    "fast_eva"       : Eva_FastSpeechInference,
    "fast_elizabeth" : Elizabeth_FastSpeechInference,

    "trans_thorsten" : Thorsten_TransformerTTSInference,
    "trans_lj"       : LJSpeech_TransformerTTSInference,
    "trans_libri"    : LibriTTS_TransformerTTSInference,
    "trans_karl"     : Karlsson_TransformerTTSInference,
    "trans_eva"      : Eva_TransformerTTSInference,
    "trans_elizabeth": Elizabeth_TransformerTTSInference
    }


def read_texts(model_id, sentence, filename, device="cpu", speaker_embedding="default_spemb.pt"):
    tts = tts_dict[model_id](device=device, speaker_embedding=speaker_embedding)
    if type(sentence) == str:
        sentence = [sentence]
    tts.read_to_file(text_list=sentence, file_location=filename)
    del tts


tl_en = """Peter Piper picked a peck of pickled peppers.
A peck of pickled peppers Peter Piper picked.
If Peter Piper picked a peck of pickled peppers, where’s the peck of pickled peppers Peter Piper picked?
Betty Botter bought some butter, but she said the butter’s bitter.
If I put it in my batter, it will make my batter bitter!
But a bit of better butter will make my batter better.
So ‘twas better Betty Botter bought a bit of better butter.
How much wood would a woodchuck chuck if a woodchuck could chuck wood?
He would chuck, he would, as much as he could, and chuck as much wood, as a woodchuck would if a woodchuck could chuck wood.
She sells seashells by the seashore.
How can a clam cram in a clean cream can?
I scream, you scream, we all scream for ice cream!
Susie works in a shoeshine shop. Where she shines she sits, and where she sits she shines.
Fuzzy Wuzzy was a bear. Fuzzy Wuzzy had no hair. Fuzzy Wuzzy wasn’t fuzzy, was he?
Can you can a can as a canner can can a can?
I have got a date at a quarter to eight; I’ll see you at the gate, so don’t be late.
You know New York, you need New York, you know you need unique New York
I saw a kitten eating chicken in the kitchen.
If a dog chews shoes, whose shoes does he choose?
I thought I thought of thinking of thanking you.
I wish to wash my Irish wristwatch.
Near an ear, a nearer ear, a nearly eerie ear.
Nine nice night nurses nursing nicely.
Wayne went to wales to watch walruses.""".split("\n")

tl_de = """"Ein Vater hatte zwei Söhne, davon war der älteste klug und gescheit, und wusste sich in alles wohl zu schicken, der jüngste aber war dumm, konnte nichts begreifen und lernen: 
und wenn ihn die Leute sahen, sprachen sie: „Mit dem wird der Vater noch seine Last haben!“ 
Wenn nun etwas zu tun war, so musste es der älteste allzeit ausrichten: hieß ihn aber der Vater noch spät oder gar in der Nacht etwas holen, und der Weg ging dabei über den Kirchhof oder sonst einen schaurigen Ort, so antwortete er wohl: 
„ach nein, Vater, ich gehe nicht dahin, es gruselt mir!“ denn er fürchtete sich. 
Oder, wenn abends beim Feuer Geschichten erzählt wurden, wobei einem die Haut schaudert, so sprachen die Zuhörer manchmal: „Ach, es gruselt mir!“ 
Der jüngste saß in einer Ecke und hörte das mit an, und konnte nicht begreifen, was es heißen sollte. „Immer sagen sie: Es gruselt mir! Es gruselt mir! 
Mir gruselt’s nicht: das wird wohl eine Kunst sein, von der ich auch nichts verstehe.“ Nun geschah es, dass der Vater einmal zu ihm sprach: 
„Hör du, in der Ecke dort, du wirst groß und stark, du musst auch etwas lernen, womit du dein Brot verdienst. 
Siehst du, wie dein Bruder sich Mühe gibt, aber an dir ist Hopfen und Malz verloren.“ 
„Ei, Vater,“ antwortete er, „ich will gerne was lernen; ja, wenns anginge, so möchte ich lernen, dass mir gruselte; davon verstehe ich noch gar nichts.“ 
Der älteste lachte, als er das hörte, und dachte bei sich: 
„Du lieber Gott, was ist mein Bruder ein Dummbart, aus dem wird sein Lebtag nichts: was ein Häkchen werden will, muss sich beizeiten krümmen.“ 
Der Vater seufzte und antwortete ihm: „Das Gruseln, das sollst du schon lernen, aber dein Brot wirst du damit nicht verdienen.“ 
Bald danach kam der Küster zum Besuch ins Haus, da klagte ihm der Vater seine Not und erzählte, wie sein jüngster Sohn in allen Dingen so schlecht beschlagen wäre, er wüsste nichts und lernte nichts. 
„Denkt Euch, als ich ihn fragte, womit er sein Brot verdienen wollte, hat er gar verlangt, das Gruseln zu lernen.“ 
„Wenn’s weiter nichts ist,“ antwortete der Küster, „das kann er bei mir lernen; tut ihn nur zu mir, ich werde ihn schon abhobeln.“ 
Der Vater war es zufrieden, weil er dachte „der Junge wird doch ein wenig zugestutzt.“ 
Der Küster nahm ihn also ins Haus, und er musste die Glocke läuten. 
Nach ein paar Tagen weckte er ihn um Mitternacht, hieß ihn aufstehen, in den Kirchturm steigen und läuten. 
„Du sollst schon lernen, was Gruseln ist,“ dachte er, ging heimlich voraus, 
und als der Junge oben war, und sich umdrehte und das Glockenseil fassen wollte, so sah er auf der Treppe, dem Schalloch gegenüber, eine weiße Gestalt stehen. 
„Wer da?“ rief er, aber die Gestalt gab keine Antwort, regte und bewegte sich nicht. 
„Gib Antwort,“ rief der Junge, „oder mache, dass du fortkommst, du hast hier in der Nacht nichts zu schaffen.“ 
Der Küster aber blieb unbeweglich stehen, damit der Junge glauben sollte, er wäre ein Gespenst. 
Der Junge rief zum zweitenmal: „Was willst du hier? Sprich, wenn du ein ehrlicher Kerl bist, oder ich werfe dich die Treppe hinab.“ 
Der Küster dachte: „Das wird so schlimm nicht gemeint sein,“ gab keinen Laut von sich und stand, als wenn er von Stein wäre. 
Da rief ihn der Junge zum drittenmal an, und als das auch vergeblich war, nahm er einen Anlauf und stieß das Gespenst die Treppe hinab, dass es zehn Stufen hinabfiel und in einer Ecke liegen blieb. 
Darauf läutete er die Glocke, ging heim, legte sich, ohne ein Wort zu sagen, ins Bett und schlief fort. 
Die Küstersfrau wartete lange Zeit auf ihren Mann, aber er wollte nicht wiederkommen. 
Da ward ihr endlich angst, sie weckte den Jungen und fragte: „Weißt du nicht, wo mein Mann geblieben ist? Er ist vor dir auf den Turm gestiegen.“ 
„Nein,“ antwortete der Junge, „aber da hat einer dem Schalloch gegenüber auf der Treppe gestanden, und weil er keine Antwort geben und auch nicht weggehen wollte, so habe ich ihn für einen Spitzbuben gehalten und hinuntergestoßen. 
Geht nur hin, so werdet Ihr sehen ob er’s gewesen ist, es sollte mir leid tun.“ 
Die Frau sprang fort und fand ihren Mann, der in einer Ecke lag und jammerte, und ein Bein gebrochen hatte. 
Sie trug ihn herab und eilte dann mit lautem Geschrei zu dem Vater des Jungen. 
„Euer Junge,“ rief sie, „hat ein großes Unglück angerichtet, meinen Mann hat er die Treppe hinabgeworfen, dass er ein Bein gebrochen hat: schafft den Taugenichts aus unserm Haus.“ 
Der Vater erschrak, kam herbeigelaufen und schalt den Jungen aus. „Was sind das für gottlose Streiche, die muss dir der Böse gegeben haben.“ 
„Vater,“ antwortete er, „hört nur an, ich bin ganz unschuldig: er stand da in der Nacht wie einer, der Böses im Sinne hat. 
Ich wusste nicht, wer’s war, und hab ihn dreimal ermahnt, zu reden oder wegzugehen. 
„Ach,“ sprach der Vater, „mit dir erleb ich nur Unglück, geh mir aus den Augen, ich will dich nicht mehr ansehen.“ 
„Ja, Vater, recht gerne, wartet nur, bis der Tag ist, da will ich ausgehen und das Gruseln lernen, so versteh ich doch eine Kunst, die mich ernähren kann.“""".split(
    "\n")

if __name__ == '__main__':
    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.isdir("audios"):
        os.makedirs("audios")

    read_texts(model_id="fast_libri",
               sentence=["Those of you who volunteered to be injected with praying mantis DNA, I've got some good news and some bad news.",
                         "Bad news is we're postponing those tests indefinitely.",
                         "Good news is we've got a much better test for you: fighting an army of mantis men.",
                         "Pick up a rifle and follow the yellow line.",
                         "You'll know when the test starts."],
               filename="audios/turret.wav",
               device=exec_device,
               speaker_embedding="turret.pt")

    read_texts(model_id="fast_libri",
               sentence=["All right, I've been thinking.",
                         "When life gives you lemons?",
                         "Don't make lemonade.",
                         "Make life take the lemons back!",
                         "Get mad!",
                         "I don't want your damn lemons!",
                         "What am I supposed to do with these?"],
               filename="audios/cave_lemons.wav",
               device=exec_device,
               speaker_embedding="cave_johnson.pt")

    read_texts(model_id="fast_libri",
               sentence=["Okay.",
                         "Look.",
                         "We both said a lot of things that you're going to regret.",
                         "But I think we can put our differences behind us.",
                         "For science.",
                         "You monster!"],
               filename="audios/glados_regret.wav",
               device=exec_device,
               speaker_embedding="glados.pt")

    read_texts(model_id="fast_lj", sentence=tl_en, filename="audios/fast_lj.wav", device=exec_device)

    read_texts(model_id="fast_thorsten", sentence=tl_de, filename="audios/fast_thorsten.wav", device=exec_device)
