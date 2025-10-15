import re
from functools import lru_cache
from typing import Callable, Optional

from .utils import get_lang_code

_MosesTokenizer = None
_spacy = None
_jieba = None
_pycantonese = None
_tokenize_khmer = None
_tokenize_lao = None
_tokenize_thai = None


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
    global _tokenize_khmer
    if _tokenize_khmer is None:
        from khmernltk import word_tokenize as _tokenize_khmer

        _tokenize_khmer = _tokenize_khmer
    return _tokenize_khmer(s, return_tokens=True)


def tokenize_cantonese(s: str) -> list[str]:
    """
    Tokenize Cantonese text using pycantonese.
    """
    global _pycantonese
    if _pycantonese is None:
        import pycantonese

        _pycantonese = pycantonese
    return _pycantonese.segment(s)


def tokenize_chinese(s: str) -> list[str]:
    """
    Tokenize Chinese text using jieba.
    """
    global _jieba
    if _jieba is None:
        import jieba

        _jieba = jieba
    return list(_jieba.cut(s))


def tokenize_thai(s: str) -> list[str]:
    """
    Tokenize Thai text using pythainlp.
    """
    global _tokenize_thai
    if _tokenize_thai is None:
        from pythainlp import word_tokenize

        _tokenize_thai = word_tokenize
    return _tokenize_thai(s)


def tokenize_lao(s: str) -> list[str]:
    """
    Tokenize Lao text using laonlp.
    """
    global _tokenize_lao
    if _tokenize_lao is None:
        from laonlp import word_tokenize

        _tokenize_lao = word_tokenize
    return _tokenize_lao(s)


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

    def tokenize(s: str) -> list[str]:
        f"""
        Tokenize {lang} text using spaCy.
        """
        return [t.text for t in tokenizer(s) if not t.is_space]

    return tokenize


@lru_cache(maxsize=None)
def get_tokenizer_lang_map() -> dict[str, Callable[[str], list[str]]]:
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
        # "kor": get_moses_tokenizer(lang="ko"),
        # "jpn": get_moses_tokenizer(lang="ja"),
        "jpn": get_spacy_tokenizer("jpn"),
        "kor": get_spacy_tokenizer("kor"),
    }
    return LANGUAGE_TO_TOKENIZER_MAP


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
    if part2_lang_code in get_tokenizer_lang_map():
        return get_tokenizer_lang_map()[part2_lang_code]
    elif default_moses:
        return get_moses_tokenizer(lang=lang_code)
    else:
        return tokenize_whitespace
