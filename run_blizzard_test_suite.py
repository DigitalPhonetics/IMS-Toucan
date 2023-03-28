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
                                            system,
                                            speaker,
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
                                  "« Au début, j’ai lancé ça pour m’amuser avec mes copains, mais si on gagne un peu d’argent en plus, c’est tant mieux », commente le jeune homme, tout sourire. L’idée lui est venue début janvier, quand les autorités ont encouragé « l’économie de la rue » pour tenter de relancer l’activité. "
                                  ]):
        tts.read_to_file(text_list=[sentence],
                         file_location=f"audios/{version}/{speaker}-{version}-Sentence{i}-{system}.wav",
                         duration_scaling_factor=duration_scaling_factor,
                         pitch_variance_scale=pitch_variance_scale,
                         energy_variance_scale=energy_variance_scale,
                         )


def phonetically_interesting_sentences_unseen(version,
                                              system,
                                              speaker,
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

    for i, sentence in enumerate([
        "On vous inviterait à chasser l'ours dans les montagnes de la Suisse, que vous diriez: «Très bien!»",
        "Le Renard s’en saisit, et dit: Mon bon Monsieur, apprenez que tout flatteur vit aux dépens de celui qui l’écoute! Cette leçon vaut bien un fromage sans doute. Le Corbeau honteux et confus jura, mais un peu tard, qu’on ne l’y prendrait plus!",
        "De nombreux membres le considéraient comme un traître de l'organisation et il a reçu plusieurs menaces de mort de la part du groupe. Depuis, les récits populaires ont largement emboîté le pas.",
        "La révolution survint, les événements se précipitèrent, les familles parlementaires décimées, chassées, traquées, se dispersèrent.",
        "Quel est ce bonhomme qui me regarde ?",
        "Mais, soit qu’il n’eût pas remarqué cette manœuvre ou qu’il n’eut osé s’y soumettre, la prière était finie que le nouveau tenait encore sa casquette sur ses deux genoux. ",
        "Il traîna l'échelle jusqu'à ce trou, dont le diamètre mesurait six pieds environ, et la laissa se dérouler, après avoir solidement attaché son extrémité supérieure.",
        "En voilà une de ces destinées qui peut se vanter d'être flatteuse! À boire, Chourineur, dit brusquement Fleur-de-Marie après un assez long silence; et elle tendit son verre."
    ]):
        tts.read_to_file(text_list=[sentence],
                         file_location=f"audios/{version}/{speaker}-{version}-Sentence{i}-{system}.mp3",
                         duration_scaling_factor=duration_scaling_factor,
                         pitch_variance_scale=pitch_variance_scale,
                         energy_variance_scale=energy_variance_scale,
                         )


if __name__ == '__main__':
    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running on {exec_device}")

    # Component 4: Sentence Embedding / Without
    phonetically_interesting_sentences_unseen(version="Component1",
                                              system="SystemA",
                                              speaker="AD",
                                              model_id="AD_finetuned_with_sentence",
                                              exec_device=exec_device,
                                              vocoder_model_path=None,
                                              biggan=True,
                                              speaker_reference="audios/blizzard_references/DIVERS_BOOK_AD_04_0001_143.wav"
                                              )

    phonetically_interesting_sentences_unseen(version="Component1",
                                              system="SystemB",
                                              speaker="AD",
                                              model_id="AD_finetuned",
                                              exec_device=exec_device,
                                              vocoder_model_path=None,
                                              biggan=True,
                                              speaker_reference="audios/blizzard_references/DIVERS_BOOK_AD_04_0001_143.wav"
                                              )

    phonetically_interesting_sentences_unseen(version="Component1",
                                              system="SystemA",
                                              speaker="NEB",
                                              model_id="NEB_finetuned_with_sentence",
                                              exec_device=exec_device,
                                              vocoder_model_path=None,
                                              biggan=True,
                                              speaker_reference="audios/blizzard_references/ES_LMP_NEB_02_0002_117.wav"
                                              )

    phonetically_interesting_sentences_unseen(version="Component1",
                                              system="SystemB",
                                              speaker="NEB",
                                              model_id="NEB_finetuned",
                                              exec_device=exec_device,
                                              vocoder_model_path=None,
                                              biggan=True,
                                              speaker_reference="audios/blizzard_references/ES_LMP_NEB_02_0002_117.wav"
                                              )
