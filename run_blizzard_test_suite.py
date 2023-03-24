import os

import torch

from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface


def le_corbeau_et_le_renard(version, model_id="Meta", exec_device="cpu", speaker_reference=None, vocoder_model_path=None, biggan=False):
    os.makedirs("audios", exist_ok=True)
    os.makedirs(f"audios/{version}", exist_ok=True)
    tts = ToucanTTSInterface(device=exec_device, tts_model_path=model_id, vocoder_model_path=vocoder_model_path, faster_vocoder=not biggan)
    tts.set_language("fr")
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)

    for i, sentence in enumerate(["Maître Corbeau, sur un arbre perché, tenait en son bec un fromage.",
                                  "Maître Renard, par l’odeur alléché, lui tint à peu près ce langage:",
                                  "Et bonjour, Monsieur du Corbeau, que vous êtes joli! que vous me semblez beau!",
                                  "Sans mentir, si votre ramage se rapporte à votre plumage, vous êtes le Phénix des hôtes de ces bois.",
                                  "À ces mots le Corbeau ne se sent pas de joie, et pour montrer sa belle voix, il ouvre un large bec, laisse tomber sa proie.",
                                  "Le Renard s’en saisit, et dit: Mon bon Monsieur, apprenez que tout flatteur vit aux dépens de celui qui l’écoute.",
                                  "Cette leçon vaut bien un fromage sans doute.",
                                  "Le Corbeau honteux et confus jura, mais un peu tard, qu’on ne l’y prendrait plus.",
                                  "Maître Corbeau, sur un arbre perché, tenait en son bec un fromage. Maître Renard, par l’odeur alléché, lui tint à peu près ce langage: Et bonjour, Monsieur du Corbeau, que vous êtes joli! que vous me semblez beau! Sans mentir, si votre ramage se rapporte à votre plumage, vous êtes le Phénix des hôtes de ces bois. À ces mots le Corbeau ne se sent pas de joie, et pour montrer sa belle voix, il ouvre un large bec, laisse tomber sa proie. Le Renard s’en saisit, et dit: Mon bon Monsieur, apprenez que tout flatteur vit aux dépens de celui qui l’écoute. Cette leçon vaut bien un fromage sans doute. Le Corbeau honteux et confus jura, mais un peu tard, qu’on ne l’y prendrait plus."
                                  ]):
        tts.read_to_file(text_list=[sentence], file_location=f"audios/{version}/Le_corbeau_et_le_renard_{i}.wav")


def phonetically_interesting_sentences_seen(version,
                                            model_id="Meta",
                                            exec_device="cpu",
                                            speaker_reference=None,
                                            vocoder_model_path=None,
                                            biggan=False,
                                            duration_scaling_factor=1.0,
                                            pitch_variance_scale=1.0,
                                            energy_variance_scale=1.0):
    os.makedirs("audios", exist_ok=True)
    os.makedirs(f"audios/{version}", exist_ok=True)
    tts = ToucanTTSInterface(device=exec_device, tts_model_path=model_id, vocoder_model_path=vocoder_model_path, faster_vocoder=not biggan)
    tts.set_language("fr")
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)

    for i, sentence in enumerate(["Les passagers, s'aidant les uns les autres, parvinrent à se dégager des mailles du filet.",
                                  "On peut le faire si ce principe, vient en contrepartie d'un autre principe de même niveau.",
                                  "Ce manuscrit, signé de mon nom, complété par l'histoire de ma vie, sera renfermé dans un petit appareil insubmersible.",
                                  "On vous inviterait à chasser l'ours dans les montagnes de la Suisse, que vous diriez: «Très bien!»",
                                  "Mais de là à monter cette histoire en épingle",
                                  "La France mérite un tout autre projet.",
                                  "Mon maître, dit alors nab, j'ai l'idée que nous pouvons chercher tant que nous voudrons le monsieur dont il s'agit, mais que nous ne le découvrirons que quand il lui plaira.",
                                  "Pendant la première semaine du mois d'août, les rafales s'apaisèrent peu à peu, et l'atmosphère recouvra un calme qu'elle semblait avoir à jamais perdu.",
                                  ]):
        tts.read_to_file(text_list=[sentence], file_location=f"audios/{version}/seen_sentences_{i}.wav",
                         duration_scaling_factor=duration_scaling_factor,
                         pitch_variance_scale=pitch_variance_scale,
                         energy_variance_scale=energy_variance_scale,
                         )


def phonetically_interesting_sentences_unseen(version,
                                              model_id="Meta",
                                              exec_device="cpu",
                                              speaker_reference=None,
                                              vocoder_model_path=None,
                                              biggan=False,
                                              duration_scaling_factor=1.0,
                                              pitch_variance_scale=1.0,
                                              energy_variance_scale=1.0):
    os.makedirs("audios", exist_ok=True)
    os.makedirs(f"audios/{version}", exist_ok=True)
    tts = ToucanTTSInterface(device=exec_device, tts_model_path=model_id, vocoder_model_path=vocoder_model_path, faster_vocoder=not biggan)
    tts.set_language("fr")
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)

    for i, sentence in enumerate([  # "Les amis ont vu un ancien ami en avril, dit-on.",
         "Des amis ont vu en avril un vieil ami qui était très aimable, dit-on.",
        "C'est une maison où l'on peut aller quand il pleut.",
        "Après un tour de présentation, ils sont allés",
        "Le village de Beaulieu est en grand émoi.",
        "Le Premier Ministre a en effet décidé de faire étape dans cette commune au cours de sa tournée de la région en fin d'année.",
        "Jusqu'ici les seuls titres de gloire de Beaulieu étaient son vin blanc sec, ses chemises en soie, un champion local de course à pied (Louis Garret), quatrième aux jeux olymiques de Berlin en 1936, et plus récemment, son usine de pâtes italiennes."
    ]):
        tts.read_to_file(text_list=[sentence],
                         file_location=f"audios/{version}/unseen_sentences_{i}.wav",
                         duration_scaling_factor=duration_scaling_factor,
                         pitch_variance_scale=pitch_variance_scale,
                         energy_variance_scale=energy_variance_scale,
                         )


if __name__ == '__main__':
    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running on {exec_device}")

    # Key:
    # [avocodo0/biggan1] _ [pretraining0/scratch1] _ [prosody unmodded0/prosody modded1] _ [with sent emb0/without1] _ [with discriminator0/without1] _ [neb/ad]

    phonetically_interesting_sentences_unseen(version="1_0_1_1_1_ad",
                                              model_id="AD_f",
                                              exec_device=exec_device,
                                              vocoder_model_path=None,
                                              biggan=True,
                                              duration_scaling_factor=1.0,
                                              pitch_variance_scale=1.2,
                                              energy_variance_scale=1.2,
                                              speaker_reference="audios/blizzard_references/DIVERS_BOOK_AD_04_0001_143.wav"
                                              )

    phonetically_interesting_sentences_unseen(version="1_0_0_1_1_ad",
                                              model_id="AD_f",
                                              exec_device=exec_device,
                                              vocoder_model_path=None,
                                              biggan=True,
                                              duration_scaling_factor=1.0,
                                              pitch_variance_scale=1.0,
                                              energy_variance_scale=1.0,
                                              speaker_reference="audios/blizzard_references/DIVERS_BOOK_AD_04_0001_143.wav"
                                              )

    phonetically_interesting_sentences_unseen(version="1_0_0_1_1_neb",
                                              model_id="NEB",
                                              exec_device=exec_device,
                                              vocoder_model_path=None,
                                              biggan=True,
                                              duration_scaling_factor=1.0,
                                              pitch_variance_scale=1.0,
                                              energy_variance_scale=1.0,
                                              speaker_reference="audios/blizzard_references/ES_LMP_NEB_02_0002_117.wav"
                                              )

    phonetically_interesting_sentences_unseen(version="1_0_1_1_1_neb",
                                              model_id="NEB",
                                              exec_device=exec_device,
                                              vocoder_model_path=None,
                                              biggan=True,
                                              duration_scaling_factor=1.0,
                                              pitch_variance_scale=1.3,
                                              energy_variance_scale=1.3,
                                              speaker_reference="audios/blizzard_references/ES_LMP_NEB_02_0002_117.wav"
                                              )
