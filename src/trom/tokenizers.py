import os
import re
from functools import lru_cache
from typing import Callable, Optional

import pkuseg
import pycantonese
import spacy
from khmernltk import word_tokenize as _tokenize_khmer
from laonlp import word_tokenize as tokenize_lao
from pythainlp import word_tokenize as _tokenize_thai
from sacremoses import MosesTokenizer

from .utils import get_lang_code


def word_tokenize(text: str) -> list[str]:
    """Tokenizes text into words using a simple regex."""
    return re.findall(r"\w+", text)


def remove_whitespaces(s: str) -> str:
    """Removes all whitespaces from a string."""
    return "".join(s.split())


def tokenizer_khmer(s: str) -> list[str]:
    """
    Tokenize Khmer text using khmernltk.
    """
    return _tokenize_khmer(s, return_tokens=True)


def tokenize_cantonese(s: str) -> list[str]:
    """
    Tokenize Cantonese text using pycantonese.
    """
    return pycantonese.segment(s)


if os.getenv("CHINESE_TOKENIZER", "pkuseg") == "pkuseg":

    @lru_cache()
    def _get_pkuseg_seg():
        return pkuseg.pkuseg()

    def tokenize_chinese(s: str) -> list[str]:
        """
        Tokenize Chinese text using pkuseg.
        """
        return _get_pkuseg_seg().cut(s)
else:

    @lru_cache()
    def _get_jieba():
        import jieba

        return jieba

    def tokenize_chinese(s: str) -> list[str]:
        """
        Tokenize Chinese text using jieba.
        """
        return list(_get_jieba().cut(s))


def tokenize_thai(s: str) -> list[str]:
    """
    Tokenize Thai text using pythainlp.
    """
    return _tokenize_thai(s)


def tokenize_whitespace(s: str) -> list[str]:
    """
    Tokenize text by whitespace.
    """
    return [t.strip() for t in re.split(r"\s+", s.strip()) if t.strip()]


def get_moses_tokenizer(lang: Optional[str] = None) -> Callable[[str], list[str]]:
    """
    Tokenize text using Moses tokenizer.
    """
    tokenizer = MosesTokenizer(lang=lang)

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
    lang = get_lang_code(lang)

    if lang not in SPACY_LANGUAGE_TO_MODEL_MAP:
        raise ValueError(f"Unsupported language: {lang}")

    try:
        spacy_model = SPACY_LANGUAGE_TO_MODEL_MAP[lang]
        nlp = spacy.load(
            spacy_model,
            disable=["ner", "parser", "entity", "tok2vec", "attribute_ruler"],
        )
    except OSError:
        spacy.cli.download(spacy_model)
        return get_spacy_tokenizer(lang)

    tokenizer = nlp.tokenizer

    def tokenize(s: str) -> list[str]:
        f"""
        Tokenize {lang} text using spaCy.
        """
        return [t.text for t in tokenizer(s) if not t.is_space]

    return tokenize


LANGUAGE_TO_TOKENIZER_MAP = {
    "khm": tokenizer_khmer,
    "yue": tokenize_cantonese,
    "zho": tokenize_chinese,
    "cmn": tokenize_chinese,  # Mandarin Chinese
    "tha": tokenize_thai,
    "lao": tokenize_lao,
    # SacreMoses: French splits "l'espoir" in ["l'", "espoir"] which is harmful for our purposes
    "fra": tokenize_whitespace,
    "cjk": get_moses_tokenizer(lang="zh"),
    "eng": get_moses_tokenizer(lang="en"),
    "kor": get_moses_tokenizer(lang="ko"),
    # "jpn": get_moses_tokenizer(lang="ja"),
    "jpn": get_spacy_tokenizer("jpn"),
    "kor": get_spacy_tokenizer("kor"),
}


def get_tokenizer_for_lang(
    lang_code: str, default_moses: bool = False
) -> Callable[[str], list[str]]:
    """
    Get a tokenizer function for the given language code.
    If no particular tokenizer is available,
    it defaults to Moses tokenizer if `default_moses` is True,
    otherwise it defaults to whitespace tokenizer.
    """
    part2_lang_code = get_lang_code(lang_code)
    if part2_lang_code in LANGUAGE_TO_TOKENIZER_MAP:
        return LANGUAGE_TO_TOKENIZER_MAP[part2_lang_code]
    elif default_moses:
        return get_moses_tokenizer(lang=lang_code)
    else:
        return tokenize_whitespace
