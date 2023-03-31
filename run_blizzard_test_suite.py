import os

import pyloudnorm as pyln
import soundfile as sf
import torch
from pedalboard import HighpassFilter
from pedalboard import LowpassFilter
from pedalboard import Pedalboard

from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface


def phonetically_interesting_sentences_unseen(version,
                                              system,
                                              model_id="Meta",
                                              device="cpu", ):
    sentences = [
        "On vous inviterait à chasser l'ours dans les montagnes de la Suisse, que vous diriez: «Très bien!»",
        "Maître Corbeau, sur un arbre perché, tenait en son bec un fromage. Maître Renard, par l’odeur alléché, lui tint à peu près ce langage: Et bonjour, Monsieur du Corbeau, que vous êtes joli! que vous me semblez beau! Sans mentir, si votre ramage se rapporte à votre plumage, vous êtes le Phénix des hôtes de ces bois. À ces mots le Corbeau ne se sent pas de joie, et pour montrer sa belle voix, il ouvre un large bec, laisse tomber sa proie. Le Renard s’en saisit, et dit: Mon bon Monsieur, apprenez que tout flatteur vit aux dépens de celui qui l’écoute. Cette leçon vaut bien un fromage sans doute. Le Corbeau honteux et confus jura, mais un peu tard, qu’on ne l’y prendrait plus."
        "De nombreux membres le considéraient comme un traître de l'organisation et il a reçu plusieurs menaces de mort de la part du groupe. Depuis, les récits populaires ont largement emboîté le pas.",
        "La révolution survint, les événements se précipitèrent, les familles parlementaires décimées, chassées, traquées, se dispersèrent.",
        "Quel est ce bonhomme qui me regarde ?",
        "Mais, soit qu’il n’eût pas remarqué cette manœuvre ou qu’il n’eut osé s’y soumettre, la prière était finie que le nouveau tenait encore sa casquette sur ses deux genoux. ",
        "Il traîna l'échelle jusqu'à ce trou, dont le diamètre mesurait six pieds environ, et la laissa se dérouler, après avoir solidement attaché son extrémité supérieure.",
        "En voilà une de ces destinées qui peut se vanter d'être flatteuse! À boire, Chourineur, dit brusquement Fleur-de-Marie après un assez long silence; et elle tendit son verre."
    ]
    read_sentences_ad(sentences,
                      version,
                      system,
                      model_id,
                      device)
    read_sentences_neb(sentences,
                       version,
                       system,
                       model_id,
                       device)


def read_sentences_ad(sentences,
                      version,
                      system,
                      model_id,
                      device="cpu"):
    os.makedirs("audios", exist_ok=True)
    os.makedirs(f"audios/{version}", exist_ok=True)
    tts = ToucanTTSInterface(device=device, tts_model_path=model_id, faster_vocoder=False)
    tts.set_language("fr")
    tts.set_utterance_embedding("audios/blizzard_references/DIVERS_BOOK_AD_04_0001_143.wav")
    effects = Pedalboard(plugins=[HighpassFilter(cutoff_frequency_hz=300),
                                  LowpassFilter(cutoff_frequency_hz=10000)])
    for i, sentence in enumerate(sentences):
        print("Now synthesizing: {}".format(sentence))
        silence = torch.zeros([2000])
        wav = tts(sentence,
                  durations=None,
                  pitch=None,
                  energy=None,
                  duration_scaling_factor=1.0,
                  pitch_variance_scale=1.0,
                  energy_variance_scale=1.0).cpu()
        wav = torch.cat((silence, wav, silence), 0)
        wav = [val for val in wav for _ in (0, 1)]
        sr = 48000
        wav = effects(wav, sr)
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, -32.0)
        sf.write(file=f"audios/{version}/AD-{version}-Sentence{i}-{system}.mp3",
                 data=wav, samplerate=sr)


def read_sentences_neb(sentences,
                       version,
                       system,
                       model_id,
                       device="cpu"):
    os.makedirs("audios", exist_ok=True)
    os.makedirs(f"audios/{version}", exist_ok=True)
    tts = ToucanTTSInterface(device=device, tts_model_path=model_id, faster_vocoder=False)
    tts.set_language("fr")
    tts.set_utterance_embedding("audios/blizzard_references/ES_LMP_NEB_02_0002_117.wav")
    effects = Pedalboard(plugins=[HighpassFilter(cutoff_frequency_hz=300),
                                  LowpassFilter(cutoff_frequency_hz=10000)])
    for i, sentence in enumerate(sentences):
        print("Now synthesizing: {}".format(sentence))
        silence = torch.zeros([2000])
        wav = tts(sentence,
                  durations=None,
                  pitch=None,
                  energy=None,
                  duration_scaling_factor=1.0,
                  pitch_variance_scale=1.2,
                  energy_variance_scale=1.2).cpu()
        wav = torch.cat((silence, wav, silence), 0)
        wav = [val for val in wav for _ in (0, 1)]
        sr = 48000
        wav = effects(wav, sr)
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, -28.0)
        sf.write(file=f"audios/{version}/NEB-{version}-Sentence{i}-{system}.mp3",
                 data=wav, samplerate=sr)


if __name__ == '__main__':
    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running on {exec_device}")

    phonetically_interesting_sentences_unseen(version="Component1",
                                              system="SystemA",
                                              model_id="AD_finetuned",
                                              device=exec_device)

    phonetically_interesting_sentences_unseen(version="Component1",
                                              system="SystemB",
                                              model_id="AD_finetuned_with_word",
                                              device=exec_device)

    phonetically_interesting_sentences_unseen(version="Component1",
                                              system="SystemA",
                                              model_id="NEB_finetuned",
                                              device=exec_device)

    phonetically_interesting_sentences_unseen(version="Component1",
                                              system="SystemB",
                                              model_id="NEB_finetuned_with_word",
                                              device=exec_device)
