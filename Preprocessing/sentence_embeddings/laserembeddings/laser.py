from typing import Dict, Any, Union, List, Optional
from io import TextIOBase, BufferedIOBase
import os

import numpy as np

from Preprocessing.sentence_embeddings.laserembeddings.preprocessing import Tokenizer, BPE, SPM
from Preprocessing.sentence_embeddings.laserembeddings.embedding import BPESentenceEmbedding, SPMSentenceEmbedding
from Preprocessing.sentence_embeddings.laserembeddings.utils import sre_performance_patch, download_models
from Utility.storage_config import MODELS_DIR

__all__ = ['Laser']


class Laser:
    """
    End-to-end LASER embedding.

    The pipeline is: ``Tokenizer.tokenize`` -> ``BPE.encode_tokens`` -> ``BPESentenceEmbedding.embed_bpe_sentences``
    Using spm model: ``Tokenizer.tokenize`` -> ``SPM.encode_sentence`` -> ``SPMSentenceEmbedding.embed_spm_sentences``

    Args:
        mode (str): spm or bpe
        bpe_codes (str or TextIOBase, optional): the path to LASER's BPE codes (``93langs.fcodes``),
            or a text-mode file object. If omitted, ``Laser.DEFAULT_BPE_CODES_FILE`` is used.
        bpe_codes (str or TextIOBase, optional): the path to LASER's BPE vocabulary (``93langs.fvocab``),
            or a text-mode file object. If omitted, ``Laser.DEFAULT_BPE_VOCAB_FILE`` is used.
        encoder (str or BufferedIOBase, optional): the path to LASER's encoder PyToch model (``bilstm.93langs.2018-12-26.pt``),
            or a binary-mode file object. If omitted, ``Laser.DEFAULT_ENCODER_FILE`` is used.
        spm_model (str or BufferedIOBase, optional): the path to LASER's SPM model
        spm_vocab (str or BufferedIOBase, optional): the path to LASER's SPM vocab
        tokenizer_options (Dict[str, Any], optional): additional arguments to pass to the tokenizer.
            See ``.preprocessing.Tokenizer``.
        embedding_options (Dict[str, Any], optional): additional arguments to pass to the embedding layer.
            See ``.embedding.BPESentenceEmbedding``.
    
    Class attributes:
        DATA_DIR (str): the path to the directory of default LASER files.
        DEFAULT_BPE_CODES_FILE: the path to default BPE codes file.
        DEFAULT_BPE_VOCAB_FILE: the path to default BPE vocabulary file.
        DEFAULT_ENCODER_FILE: the path to default LASER encoder PyTorch model file.
    """

    DATA_DIR = os.path.join(MODELS_DIR, 'Language_Models')
    DEFAULT_BPE_CODES_FILE = os.path.join(DATA_DIR, '93langs.fcodes')
    DEFAULT_BPE_VOCAB_FILE = os.path.join(DATA_DIR, '93langs.fvocab')
    DEFAULT_ENCODER_LASER_FILE = os.path.join(DATA_DIR,
                                        'bilstm.93langs.2018-12-26.pt')
    DEFAULT_ENCODER_LASER2_FILE = os.path.join(DATA_DIR, 'laser2.pt')
    DEFAULT_SPM_MODEL_FILE = os.path.join(DATA_DIR, 'laser2.spm')
    DEFAULT_SPM_VOCAB_FILE = os.path.join(DATA_DIR, 'laser2.cvocab')

    def __init__(self,
                 mode: str = 'spm',
                 bpe_codes: Optional[Union[str, TextIOBase]] = None,
                 bpe_vocab: Optional[Union[str, TextIOBase]] = None,
                 encoder: Optional[Union[str, BufferedIOBase]] = None,
                 spm_model: Optional[Union[str, BufferedIOBase]] = None,
                 spm_vocab: Optional[Union[str, BufferedIOBase]] = None,
                 tokenizer_options: Optional[Dict[str, Any]] = None,
                 embedding_options: Optional[Dict[str, Any]] = None):

        if tokenizer_options is None:
            tokenizer_options = {}
        if embedding_options is None:
            embedding_options = {}
        
        self.bpe = None
        self.spm = None

        if mode == 'bpe':
            if bpe_codes is None:
                if not os.path.isfile(self.DEFAULT_BPE_CODES_FILE):
                    download_models(self.DATA_DIR, version=1)
                bpe_codes = self.DEFAULT_BPE_CODES_FILE
            if bpe_vocab is None:
                if not os.path.isfile(self.DEFAULT_BPE_VOCAB_FILE):
                    download_models(self.DATA_DIR, version=1)
                bpe_vocab = self.DEFAULT_BPE_VOCAB_FILE
            if encoder is None:
                if not os.path.isfile(self.DEFAULT_ENCODER_LASER_FILE):
                    download_models(self.DATA_DIR, version=1)
                encoder = self.DEFAULT_ENCODER_LASER_FILE
            
            print("Mode BPE")
            print("Using encoder: {}".format(encoder))

            self.tokenizer_options = tokenizer_options
            self.tokenizers: Dict[str, Tokenizer] = {}

            self.bpe = BPE(bpe_codes, bpe_vocab)
            self.bpeSentenceEmbedding = BPESentenceEmbedding(
                encoder, **embedding_options)
        
        if mode == 'spm':
            if spm_model is None:
                if not os.path.isfile(self.DEFAULT_SPM_MODEL_FILE):
                    download_models(self.DATA_DIR, version=2)
                spm_model = self.DEFAULT_SPM_MODEL_FILE
            if spm_vocab is None:
                if not os.path.isfile(self.DEFAULT_SPM_VOCAB_FILE):
                    download_models(self.DATA_DIR, version=2)
                spm_vocab = self.DEFAULT_SPM_VOCAB_FILE
            if encoder is None:
                if not os.path.isfile(self.DEFAULT_ENCODER_LASER2_FILE):
                    download_models(self.DATA_DIR, version=2)
                encoder = self.DEFAULT_ENCODER_LASER2_FILE
            
            print("Mode SPM")
            print("Using encoder: {}".format(encoder))

            self.tokenizer_options = tokenizer_options
            self.tokenizers: Dict[str, Tokenizer] = {}

            self.spm = SPM(spm_model)
            self.spmSentenceEmbedding = SPMSentenceEmbedding(
                encoder, spm_vocab=spm_vocab, **embedding_options)

    def _get_tokenizer(self, lang: str) -> Tokenizer:
        """Returns the Tokenizer instance for the specified language. The returned tokenizers are cached."""

        if lang not in self.tokenizers:
            self.tokenizers[lang] = Tokenizer(lang, **self.tokenizer_options)

        return self.tokenizers[lang]

    def embed_sentences(self, sentences: Union[List[str], str],
                        lang: Union[str, List[str]]="en") -> np.ndarray:
        """
        Computes the LASER embeddings of provided sentences using the tokenizer for the specified language.

        Args:
            sentences (str or List[str]): the sentences to compute the embeddings from.
            lang (str or List[str]): the language code(s) (ISO 639-1) used to tokenize the sentences
                (either as a string - same code for every sentence - or as a list of strings - one code per sentence).

        Returns:
            np.ndarray: A N * 1024 NumPy array containing the embeddings, N being the number of sentences provided.
        """
        sentences = [sentences] if isinstance(sentences, str) else sentences
        lang = [lang] * len(sentences) if isinstance(lang, str) else lang

        if len(sentences) != len(lang):
            raise ValueError(
                'lang: invalid length: the number of language codes does not match the number of sentences'
            )

        with sre_performance_patch():  # see https://bugs.python.org/issue37723
            if self.bpe:
                sentence_tokens = [
                self._get_tokenizer(sentence_lang).tokenize(sentence)
                for sentence, sentence_lang in zip(sentences, lang)
                ]
                bpe_encoded = [
                    self.bpe.encode_tokens(tokens) for tokens in sentence_tokens
                ]
                return self.bpeSentenceEmbedding.embed_bpe_sentences(bpe_encoded)
            if self.spm:
                spm_encoded = [
                    self.spm.encode_sentence(sentence) for sentence in sentences
                ]
                return self.spmSentenceEmbedding.embed_spm_sentences(spm_encoded)