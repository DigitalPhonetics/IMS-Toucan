from typing import Union, Optional
from io import TextIOBase

from sacremoses import MosesPunctNormalizer, MosesTokenizer
from sacremoses.util import xml_unescape
from subword_nmt.apply_bpe import BPE as subword_nmt_bpe, read_vocabulary
from transliterate import translit
import sentencepiece

from Preprocessing.sentence_embeddings.laserembeddings.utils import adapt_bpe_codes

# Extras
try:
    import jieba
    jieba.setLogLevel(60)
except ImportError:
    jieba = None

try:
    import MeCab
    import ipadic
except ImportError:
    MeCab = None

__all__ = ['Tokenizer', 'BPE']

###############################################################################
#
# Tokenizer
#
###############################################################################


class Tokenizer:
    """
    Tokenizer.

    Args:
        lang (str): the language code (ISO 639-1) of the texts to tokenize
        lower_case (bool, optional): if True, the texts are lower-cased before being tokenized.
            Defaults to True.
        romanize (bool or None, optional): if True, the texts are romanized.
            Defaults to None (romanization enabled based on input language).
        descape (bool, optional): if True, the XML-escaped symbols get de-escaped.
            Default to False.
    """

    def __init__(self,
                 lang: str = 'en',
                 lower_case: bool = True,
                 romanize: Optional[bool] = None,
                 descape: bool = False):
        assert lower_case, 'lower case is needed by all the models'

        if lang in ('cmn', 'wuu', 'yue'):
            lang = 'zh'
        if lang == 'jpn':
            lang = 'ja'

        if lang == 'zh' and jieba is None:
            raise ModuleNotFoundError(
                '''No module named 'jieba'. Install laserembeddings with 'zh' extra to fix that: "pip install laserembeddings[zh]"'''
            )
        if lang == 'ja' and MeCab is None:
            raise ModuleNotFoundError(
                '''No module named 'MeCab'. Install laserembeddings with 'ja' extra to fix that: "pip install laserembeddings[ja]"'''
            )

        self.lang = lang
        self.lower_case = lower_case
        self.romanize = romanize if romanize is not None else lang == 'el'
        self.descape = descape

        self.normalizer = MosesPunctNormalizer(lang=lang)
        self.tokenizer = MosesTokenizer(lang=lang)
        self.mecab_tokenizer = MeCab.Tagger(
            f"{ipadic.MECAB_ARGS} -Owakati -b 50000") if lang == 'ja' else None

    def tokenize(self, text: str) -> str:
        """Tokenizes a text and returns the tokens as a string"""

        # REM_NON_PRINT_CHAR
        # not implemented

        # NORM_PUNC
        text = self.normalizer.normalize(text)

        # DESCAPE
        if self.descape:
            text = xml_unescape(text)

        # MOSES_TOKENIZER
        # see: https://github.com/facebookresearch/LASER/issues/55#issuecomment-480881573
        text = self.tokenizer.tokenize(text,
                                       return_str=True,
                                       escape=False,
                                       aggressive_dash_splits=False)

        # jieba
        if self.lang == 'zh':
            text = ' '.join(jieba.cut(text.rstrip('\r\n')))

        # MECAB
        if self.lang == 'ja':
            text = self.mecab_tokenizer.parse(text).rstrip('\r\n')

        # ROMAN_LC
        if self.romanize:
            text = translit(text, self.lang, reversed=True)

        if self.lower_case:
            text = text.lower()

        return text


###############################################################################
#
# Apply BPE
#
###############################################################################


class BPE:
    """
    BPE encoder.

    Args:
        bpe_codes (str or TextIOBase): the path to LASER's BPE codes (``93langs.fcodes``),
            or a text-mode file object.
        bpe_codes (str or TextIOBase): the path to LASER's BPE vocabulary (``93langs.fvocab``),
            or a text-mode file object.
    """

    def __init__(self, bpe_codes: Union[str, TextIOBase],
                 bpe_vocab: Union[str, TextIOBase]):

        f_bpe_codes = None
        f_bpe_vocab = None

        try:
            if isinstance(bpe_codes, str):
                f_bpe_codes = open(bpe_codes, 'r', encoding='utf-8')  # pylint: disable=consider-using-with
            if isinstance(bpe_vocab, str):
                f_bpe_vocab = open(bpe_vocab, 'r', encoding='utf-8')  # pylint: disable=consider-using-with

            self.bpe = subword_nmt_bpe(codes=adapt_bpe_codes(f_bpe_codes
                                                             or bpe_codes),
                                       vocab=read_vocabulary(f_bpe_vocab
                                                             or bpe_vocab,
                                                             threshold=None))
            self.bpe.version = (0, 2)

        finally:
            if f_bpe_codes:
                f_bpe_codes.close()
            if f_bpe_vocab:
                f_bpe_vocab.close()

    def encode_tokens(self, sentence_tokens: str) -> str:
        """Returns the BPE-encoded sentence from a tokenized sentence"""
        return self.bpe.process_line(sentence_tokens)

###############################################################################
#
# Apply SPM
#
###############################################################################

class SPM:
    def __init__(self, spm_model: Union[str, TextIOBase]): 
        self.spm = None
        try:
            if isinstance(spm_model, str):
                self.spm = sentencepiece.SentencePieceProcessor(model_file=spm_model)
        except FileNotFoundError:
            pass
    
    def encode_sentence(self, sentence: str) -> str:
        # NORM_PUNC + LC
        normalizer = MosesPunctNormalizer(lang="en")
        sentence = normalizer.normalize(sentence)
        sentence = sentence.lower()

        pieces = self.spm.encode_as_pieces(sentence)
        return ' '.join(pieces)