import logging
import re
from functools import lru_cache
from typing import Callable, Optional

from .environ import (
    CHINESE_TOKENIZER_BACKEND,
    JAPANESE_TOKENIZER_BACKEND,
    TOKENIZER_CACHE_SIZE,
)
from .unicode import split_utf8_near_whitespace
from .utils import get_lang_code

logger = logging.getLogger(__name__)

_MosesTokenizer = None
_spacy = None
_pycantonese = None
_tokenize_khmer = None
_tokenize_lao = None
_tokenize_thai = None
_sudachipy_tokenizer = None

SUDACHIPY_MAX_BYTES = 49149 - 1000


def word_tokenize(text: str) -> list[str]:
    """Tokenizes text into words using a simple regex."""
    return re.findall(r"\w+", text)


def remove_whitespaces(s: str) -> str:
    """Removes all whitespaces from a string."""
    return "".join(s.split())


@lru_cache(maxsize=TOKENIZER_CACHE_SIZE)
def tokenizer_khmer(s: str) -> list[str]:
    """
    Tokenize Khmer text using khmernltk.
    """
    global _tokenize_khmer
    if _tokenize_khmer is None:
        from khmernltk import word_tokenize as _tokenize_khmer
    return _tokenize_khmer(s, return_tokens=True)


@lru_cache(maxsize=TOKENIZER_CACHE_SIZE)
def tokenize_japanese(s: str) -> list[str]:
    """
    Tokenize Japanese text using SudachiPy.
    """

    def _tokenize(s: str) -> list[str]:
        global _sudachipy_tokenizer
        if _sudachipy_tokenizer is None:
            from sudachipy import SplitMode, dictionary

            _sudachipy_tokenizer = dictionary.Dictionary().create(mode=SplitMode.A)
        tokens = _sudachipy_tokenizer.tokenize(s)
        return [m.surface() for m in tokens]

    segments = split_utf8_near_whitespace(s, max_bytes=SUDACHIPY_MAX_BYTES - 100)
    tokens = []
    for segment in segments:
        tokens.extend(_tokenize(segment))
    return tokens


@lru_cache(maxsize=TOKENIZER_CACHE_SIZE)
def tokenize_cantonese(s: str) -> list[str]:
    """
    Tokenize Cantonese text using pycantonese.
    """
    global _pycantonese
    if _pycantonese is None:
        import pycantonese

        _pycantonese = pycantonese
    return _pycantonese.segment(s)


if CHINESE_TOKENIZER_BACKEND == "jieba":
    logger.debug("Using jieba for Chinese tokenization.")

    _jieba = None

    @lru_cache(maxsize=TOKENIZER_CACHE_SIZE)
    def tokenize_chinese(s: str) -> list[str]:
        """
        Tokenize Chinese text using jieba.
        """
        global _jieba
        if _jieba is None:
            import jieba

            # Suppress jieba logging
            jieba.setLogLevel(20)

            _jieba = jieba
        return list(_jieba.cut(s))

elif CHINESE_TOKENIZER_BACKEND == "thulac":
    logger.debug("Using thulac for Chinese tokenization.")
    _thulac = None

    @lru_cache(maxsize=TOKENIZER_CACHE_SIZE)
    def tokenize_chinese(s: str) -> list[str]:
        """
        Tokenize Chinese text using thulac.
        """
        global _thulac
        if _thulac is None:
            import thulac

            _thulac = thulac.thulac(seg_only=True)
        tokens = _thulac.cut(s)
        return [token[0] for token in tokens]
else:
    logger.debug("Using pkuseg for Chinese tokenization.")
    _pkuseg = None

    @lru_cache(maxsize=TOKENIZER_CACHE_SIZE)
    def tokenize_chinese(s: str) -> list[str]:
        """
        Tokenize Chinese text using pkuseg.
        """
        global _pkuseg
        if _pkuseg is None:
            import pkuseg

            _pkuseg = pkuseg.pkuseg()
        return _pkuseg.cut(s)


@lru_cache(maxsize=TOKENIZER_CACHE_SIZE)
def tokenize_thai(s: str) -> list[str]:
    """
    Tokenize Thai text using pythainlp.
    """
    global _tokenize_thai
    if _tokenize_thai is None:
        from pythainlp import word_tokenize

        _tokenize_thai = word_tokenize
    return _tokenize_thai(s)


@lru_cache(maxsize=TOKENIZER_CACHE_SIZE)
def tokenize_lao(s: str) -> list[str]:
    """
    Tokenize Lao text using laonlp.
    """
    global _tokenize_lao
    if _tokenize_lao is None:
        from laonlp import word_tokenize

        _tokenize_lao = word_tokenize
    return _tokenize_lao(s)


@lru_cache(maxsize=TOKENIZER_CACHE_SIZE)
def tokenize_whitespace(s: str) -> list[str]:
    """
    Tokenize text by whitespace.
    """
    return [t.strip() for t in re.split(r"\s+", s.strip()) if t.strip()]


def get_moses_tokenizer(lang: Optional[str] = None) -> Callable[[str], list[str]]:
    """
    Tokenize text using Moses tokenizer.
    """
    global _MosesTokenizer
    if _MosesTokenizer is None:
        from sacremoses import MosesTokenizer

        _MosesTokenizer = MosesTokenizer

    tokenizer = _MosesTokenizer(lang=lang)

    @lru_cache(maxsize=TOKENIZER_CACHE_SIZE)
    def tokenize(s: str) -> list[str]:
        return tokenizer.tokenize(s, escape=False)

    return tokenize


SPACY_LANGUAGE_TO_MODEL_MAP = {
    "jpn": "ja_core_news_sm",
    "kor": "ko_core_news_sm",
    "cmn": "zh_core_web_sm",  # Mandarin Chinese
    "zho": "zh_core_web_sm",  # Simplified Chinese
}


def get_spacy_tokenizer(lang: str):
    """
    Get a spaCy tokenizer for the specified language.
    """
    global _spacy
    if _spacy is None:
        import spacy

        _spacy = spacy

    lang = get_lang_code(lang)

    if lang not in SPACY_LANGUAGE_TO_MODEL_MAP:
        raise ValueError(f"Unsupported language: {lang}")

    try:
        spacy_model = SPACY_LANGUAGE_TO_MODEL_MAP[lang]
        nlp = _spacy.load(
            spacy_model,
            disable=["ner", "parser", "entity", "tok2vec", "attribute_ruler"],
        )
    except OSError:
        _spacy.cli.download(spacy_model)
        return get_spacy_tokenizer(lang)

    tokenizer = nlp.tokenizer

    if lang == "jpn":
        # Spacy uses SudachiPy for Japanese tokenization
        # which has a max byte limit we need to respect
        @lru_cache(maxsize=TOKENIZER_CACHE_SIZE)
        def tokenize(s: str) -> list[str]:
            """Tokenize text using spaCy with SudachiPy max bytes handling."""
            segments = split_utf8_near_whitespace(
                s, max_bytes=SUDACHIPY_MAX_BYTES - 100
            )
            tokens = []
            for segment in segments:
                tokens.extend([t.text for t in nlp(segment) if not t.is_space])
            return tokens

        return tokenize
    else:

        @lru_cache(maxsize=TOKENIZER_CACHE_SIZE)
        def tokenize(s: str) -> list[str]:
            """Tokenize text using spaCy."""
            return [t.text for t in tokenizer(s) if not t.is_space]

    return tokenize


# Maps language codes to zero-argument factories so that expensive tokenizers
# (spaCy models, Moses instances, ...) are only constructed when actually requested.
_TOKENIZER_FACTORIES: dict[str, Callable[[], Callable[[str], list[str]]]] = {
    "khm": lambda: tokenizer_khmer,
    "yue": lambda: tokenize_cantonese,
    "zho": lambda: tokenize_chinese,
    "cmn": lambda: tokenize_chinese,  # Mandarin Chinese
    "tha": lambda: tokenize_thai,
    "lao": lambda: tokenize_lao,
    # SacreMoses: French splits "l'espoir" in ["l'", "espoir"] which is harmful for our purposes
    "fra": lambda: tokenize_whitespace,
    "cjk": lambda: get_moses_tokenizer(lang="zh"),
    "eng": lambda: get_moses_tokenizer(lang="en"),
    "jpn": lambda: tokenize_japanese
    if JAPANESE_TOKENIZER_BACKEND == "sudachipy"
    else get_spacy_tokenizer("jpn"),
    "kor": lambda: get_spacy_tokenizer("kor"),
}


@lru_cache(maxsize=None)
def _resolve_tokenizer(part3_lang_code: str) -> Callable[[str], list[str]]:
    return _TOKENIZER_FACTORIES[part3_lang_code]()


@lru_cache(maxsize=None)
def get_tokenizer_lang_map() -> dict[str, Callable[[str], list[str]]]:
    """Resolve and return all language-specific tokenizers at once.

    Note: this eagerly constructs every tokenizer (including spaCy models).
    Prefer `get_tokenizer_for_lang` which only loads what is requested.
    """
    return {lang: _resolve_tokenizer(lang) for lang in _TOKENIZER_FACTORIES}


def get_tokenizer_for_lang(
    lang_code: str, default_moses: bool = False
) -> Callable[[str], list[str]]:
    """
    Get a tokenizer function for the given language code.
    If no particular tokenizer is available,
    it defaults to Moses tokenizer if `default_moses` is True,
    otherwise it defaults to whitespace tokenizer.
    """
    part3_lang_code = get_lang_code(lang_code)
    if part3_lang_code in _TOKENIZER_FACTORIES:
        return _resolve_tokenizer(part3_lang_code)
    elif default_moses:
        return get_moses_tokenizer(lang=lang_code)
    else:
        return tokenize_whitespace
