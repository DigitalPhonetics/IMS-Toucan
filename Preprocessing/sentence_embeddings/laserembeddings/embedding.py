from typing import Optional, List, Union
from io import BufferedIOBase

import numpy as np

from Preprocessing.sentence_embeddings.laserembeddings.encoder import SentenceEncoder

__all__ = ['BPESentenceEmbedding']


class BPESentenceEmbedding:
    """
    LASER embeddings computation from BPE-encoded sentences.

    Args:
        encoder (str or BufferedIOBase): the path to LASER's encoder PyTorch model,
            or a binary-mode file object.
        max_sentences (int, optional): see ``.encoder.SentenceEncoder``.
        max_tokens (int, optional): see ``.encoder.SentenceEncoder``.
        stable (bool, optional): if True, mergesort sorting algorithm will be used,
            otherwise quicksort will be used. Defaults to False. See ``.encoder.SentenceEncoder``.
        cpu (bool, optional): if True, forces the use of the CPU even a GPU is available. Defaults to False.
    """

    def __init__(self,
                 encoder: Union[str, BufferedIOBase],
                 max_sentences: Optional[int] = None,
                 max_tokens: Optional[int] = 12000,
                 stable: bool = False,
                 cpu: bool = False):

        self.encoder = SentenceEncoder(
            encoder,
            max_sentences=max_sentences,
            max_tokens=max_tokens,
            sort_kind='mergesort' if stable else 'quicksort',
            cpu=cpu)

    def embed_bpe_sentences(self, bpe_sentences: List[str]) -> np.ndarray:
        """
        Computes the LASER embeddings of BPE-encoded sentences

        Args:
            bpe_sentences (List[str]): The list of BPE-encoded sentences

        Returns:
            np.ndarray: A N * 1024 NumPy array containing the embeddings, N being the number of sentences provided.
        """
        return self.encoder.encode_sentences(bpe_sentences)

class SPMSentenceEmbedding:
    """
    LASER embeddings computation from SPM-encoded sentences.

    Args:
        encoder (str or BufferedIOBase): the path to LASER's encoder PyTorch model,
            or a binary-mode file object.
        max_sentences (int, optional): see ``.encoder.SentenceEncoder``.
        max_tokens (int, optional): see ``.encoder.SentenceEncoder``.
        stable (bool, optional): if True, mergesort sorting algorithm will be used,
            otherwise quicksort will be used. Defaults to False. See ``.encoder.SentenceEncoder``.
        cpu (bool, optional): if True, forces the use of the CPU even a GPU is available. Defaults to False.
    """

    def __init__(self,
                 encoder: Union[str, BufferedIOBase],
                 spm_vocab: Union[str, BufferedIOBase],
                 max_sentences: Optional[int] = None,
                 max_tokens: Optional[int] = 12000,
                 stable: bool = False,
                 cpu: bool = False):

        self.encoder = SentenceEncoder(
            model_path=encoder,
            spm_vocab=spm_vocab,
            max_sentences=max_sentences,
            max_tokens=max_tokens,
            sort_kind='mergesort' if stable else 'quicksort',
            cpu=cpu)

    def embed_spm_sentences(self, spm_sentences: List[str]) -> np.ndarray:
        """
        Computes the LASER embeddings of SPM-encoded sentences

        Args:
            spm_sentences (List[str]): The list of SPM-encoded sentences

        Returns:
            np.ndarray: A N * 1024 NumPy array containing the embeddings, N being the number of sentences provided.
        """
        return self.encoder.encode_sentences(spm_sentences)