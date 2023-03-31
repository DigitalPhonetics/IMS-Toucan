import os

import pyloudnorm as pyln
import soundfile as sf
import torch
from pedalboard import HighpassFilter
from pedalboard import LowpassFilter
from pedalboard import Pedalboard
import numpy
from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface


def read_sentences_ad(sentences,
                      version,
                      system,
                      model_id,
                      device="cpu"):
    os.makedirs("audios", exist_ok=True)
    os.makedirs(f"audios/{version}", exist_ok=True)
    tts = ToucanTTSInterface(device=device, tts_model_path=model_id, embedding_model_path="Models/ToucanTTS_AD_finetuned_final/embedding_function.pt", faster_vocoder=False)
    tts.set_language("fr")
    tts.set_utterance_embedding("audios/blizzard_references/DIVERS_BOOK_AD_04_0001_143.wav")
    effects = Pedalboard(plugins=[HighpassFilter(cutoff_frequency_hz=100),
                                  LowpassFilter(cutoff_frequency_hz=8000)])
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
        wav = torch.cat((silence, wav, silence), 0).cpu().numpy()
        wav = numpy.array([val for val in wav for _ in (0, 1)])
        sr = 48000
        wav = effects(wav, sr)
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, -32.0)
        sf.write(file=f"audios/{version}/AD-{version}-Sentence{i}-{system}.wav",
                 data=wav, samplerate=sr)


def read_sentences_neb(sentences,
                       version,
                       system,
                       model_id,
                       device="cpu"):
    os.makedirs("audios", exist_ok=True)
    os.makedirs(f"audios/{version}", exist_ok=True)
    tts = ToucanTTSInterface(device=device, tts_model_path=model_id, embedding_model_path="Models/ToucanTTS_NEB_finetuned_final/embedding_function.pt", faster_vocoder=False)
    tts.set_language("fr")
    tts.set_utterance_embedding("audios/blizzard_references/ES_LMP_NEB_02_0002_117.wav")
    effects = Pedalboard(plugins=[HighpassFilter(cutoff_frequency_hz=100),
                                  LowpassFilter(cutoff_frequency_hz=8000)])
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
        wav = torch.cat((silence, wav, silence), 0).cpu().numpy()
        wav = numpy.array([val for val in wav for _ in (0, 1)])
        sr = 48000
        wav = effects(wav, sr)
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, -28.0)
        sf.write(file=f"audios/{version}/NEB-{version}-Sentence{i}-{system}.wav",
                 data=wav, samplerate=sr)


if __name__ == '__main__':
    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running on {exec_device}")

    sentences = ["Depuis plus de dix ans, l’E3 est considéré comme en perte de vitesse, en premier lieu parce que les géants du secteur y présentent de moins en moins leurs nouveautés.",
                 "Elle pense encore possible de ramener Robert à l'honnêteté.",
                 # "Maître Corbeau, sur un arbre perché, tenait en son bec un fromage. Maître Renard, par l’odeur alléché, lui tint à peu près ce langage.",
                 "Il avait les cheveux coupés droit sur le front, comme un chantre de village, l’air raisonnable et fort embarrassé.",
                 "Maître Corbeau, sur un arbre perché, tenait en son bec un fromage. Maître Renard, par l’odeur alléché, lui tint à peu près ce langage: Et bonjour, Monsieur du Corbeau, que vous êtes joli! que vous me semblez beau! Sans mentir, si votre ramage se rapporte à votre plumage, vous êtes le Phénix des hôtes de ces bois. À ces mots le Corbeau ne se sent pas de joie, et pour montrer sa belle voix, il ouvre un large bec, laisse tomber sa proie. Le Renard s’en saisit, et dit: Mon bon Monsieur, apprenez que tout flatteur vit aux dépens de celui qui l’écoute. Cette leçon vaut bien un fromage sans doute. Le Corbeau honteux et confus jura, mais un peu tard, qu’on ne l’y prendrait plus."
                 ]

    read_sentences_ad(version="Component1",
                      system="SystemA",
                      model_id="AD_finetuned",
                      device=exec_device,
                      sentences=sentences)

    read_sentences_ad(version="Component1",
                      system="SystemB",
                      model_id="AD_finetuned_with_word",
                      device=exec_device,
                      sentences=sentences)

    read_sentences_neb(version="Component1",
                       system="SystemA",
                       model_id="NEB_finetuned",
                       device=exec_device,
                       sentences=sentences)

    read_sentences_neb(version="Component1",
                       system="SystemB",
                       model_id="NEB_finetuned_with_word",
                       device=exec_device,
                       sentences=sentences)
