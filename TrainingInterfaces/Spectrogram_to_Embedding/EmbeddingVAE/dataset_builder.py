import os

import torch
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from TrainingInterfaces.Spectrogram_to_Embedding.StyleEmbedding import StyleEmbedding
from Utility.corpus_preparation import prepare_fastspeech_corpus


def build_speaker_embed_dataset():
    device = "cuda:5"
    embed = StyleEmbedding()
    check_dict = torch.load("Models/Embedding/embedding_function.pt", map_location="cpu")
    embed.load_state_dict(check_dict["style_emb_func"])
    embed.to(device)
    datasets = list()

    datasets.append(prepare_fastspeech_corpus(transcript_dict={},
                                              corpus_dir=os.path.join("Corpora", "Nancy"),
                                              lang="en"))

    datasets.append(prepare_fastspeech_corpus(transcript_dict={},
                                              corpus_dir=os.path.join("Corpora", "ravdess"),
                                              lang="en",
                                              ctc_selection=False))

    datasets.append(prepare_fastspeech_corpus(transcript_dict={},
                                              corpus_dir=os.path.join("Corpora", "esds"),
                                              lang="en",
                                              ctc_selection=False))

    datasets.append(prepare_fastspeech_corpus(transcript_dict={},
                                              corpus_dir=os.path.join("Corpora", "ravdess"),
                                              lang="en",
                                              ctc_selection=False))

    datasets.append(prepare_fastspeech_corpus(transcript_dict={},
                                              corpus_dir=os.path.join("Corpora", "esds"),
                                              lang="en",
                                              ctc_selection=False))

    datasets.append(prepare_fastspeech_corpus(transcript_dict={},
                                              corpus_dir=os.path.join("Corpora", "ravdess"),
                                              lang="en",
                                              ctc_selection=False))

    datasets.append(prepare_fastspeech_corpus(transcript_dict={},
                                              corpus_dir=os.path.join("Corpora", "esds"),
                                              lang="en",
                                              ctc_selection=False))

    datasets.append(prepare_fastspeech_corpus(transcript_dict={},
                                              corpus_dir=os.path.join("Corpora", "libri_all_clean"),
                                              lang="en"))

    datasets.append(prepare_fastspeech_corpus(transcript_dict={},
                                              corpus_dir=os.path.join("Corpora", "mls_porto"),
                                              lang="en"))

    datasets.append(prepare_fastspeech_corpus(transcript_dict={},
                                              corpus_dir=os.path.join("Corpora", "mls_spanish"),
                                              lang="en"))

    datasets.append(prepare_fastspeech_corpus(transcript_dict={},
                                              corpus_dir=os.path.join("Corpora", "mls_polish"),
                                              lang="en"))

    datasets.append(prepare_fastspeech_corpus(transcript_dict={},
                                              corpus_dir=os.path.join("Corpora", "mls_italian"),
                                              lang="en"))

    datasets.append(prepare_fastspeech_corpus(transcript_dict={},
                                              corpus_dir=os.path.join("Corpora", "mls_french"),
                                              lang="en"))

    datasets.append(prepare_fastspeech_corpus(transcript_dict={},
                                              corpus_dir=os.path.join("Corpora", "mls_dutch"),
                                              lang="en"))

    datasets.append(prepare_fastspeech_corpus(transcript_dict={},
                                              corpus_dir=os.path.join("Corpora", "hui_others"),
                                              lang="en"))

    train_set = ConcatDataset(datasets)

    with torch.inference_mode():
        embs = list()
        for i in tqdm(range(len(train_set))):
            spec = train_set[i][2].unsqueeze(0).clone().to(device)
            spec_len = train_set[i][3].unsqueeze(0).clone().to(device)
            emb = embed(batch_of_spectrograms=spec,
                        batch_of_spectrogram_lengths=spec_len,
                        return_only_refs=True).to("cpu").squeeze()
            embs.append(emb)

    torch.save(embs, "reference_embeddings_as_list.pt")
