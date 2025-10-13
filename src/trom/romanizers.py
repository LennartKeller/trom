import os
import re
import subprocess
import sys
import unicodedata
from functools import lru_cache
from typing import Callable, Literal, Optional

import uroman as ur
from uroman_rs import Uroman as UromanRS

from .tokenizers import get_tokenizer_for_lang
from .unicode import is_latin, string_is_only_punct

TROM_CACHE_SIZE = int(os.getenv("TROM_CACHE_SIZE", "10000000"))

JOINER_PUNCTUATION = set(":=/\\")


def get_spacing_processor(
    tokenizer: Callable[[str], list[str]],
) -> Callable[[str], str]:
    """Returns a spacing processor that uses the provided tokenizer to add spaces around punctuation.

    Args:
        tokenizer (Callable[[str], list[str]]): Tokenizer function to split the input string into words.

    Returns:
        str: Input string with spaces added around punctuation.
    """

    def spacing_processor(text: str) -> str:
        words = tokenizer(text)
        if not words:
            return text

        def string_is_only_digits(s: str) -> bool:
            return all(char.isdigit() for char in s)

        def is_joiner(s: str) -> bool:
            return all(char in JOINER_PUNCTUATION for char in s)

        result = []
        for i, token in enumerate(words):
            is_punct = string_is_only_punct(token)
            is_digit = string_is_only_digits(token)

            if is_punct:
                # Check if previous token was also punctuation
                prev_is_punct = i > 0 and string_is_only_punct(words[i - 1])

                if prev_is_punct and is_joiner(token):
                    # Joiner punctuation: attach to previous token without space
                    result.append(token)

                if prev_is_punct:
                    # Multiple consecutive punctuation: attach to previous
                    result.append(token)
                else:
                    # First punctuation after word: attach to previous word
                    result.append(token)
            elif is_digit:
                # Digits: add space before unless it's first token
                if i == 0:
                    result.append(token)
                else:
                    result.append(" " + token)
            else:
                # Regular word: add space before unless it's first token
                if i == 0:
                    result.append(token)
                else:
                    result.append(" " + token)

        return "".join(result).strip()

    return spacing_processor


def get_spacing_processor_for_lang(lang: str) -> Callable[[str], str]:
    tokenizer = get_tokenizer_for_lang(lang)
    return get_spacing_processor(tokenizer)


uroman = ur.Uroman()


@lru_cache(maxsize=TROM_CACHE_SIZE)
def uromanize(s: str) -> str:
    return uroman.romanize_string(s)


uroman_rs = UromanRS()


@lru_cache(maxsize=TROM_CACHE_SIZE)
def uromanize_rs(s: str) -> str:
    return uroman_rs.romanize(s)


class URomanizer:
    def __init__(
        self,
        preprocessor: Optional[Callable[[str], str]] = None,
        postprocessor: Optional[Callable[[str], str]] = None,
        sep: str = " ",
        split_alphanumeric: bool = False,
        backend: Literal["python", "rust"] = "python",
    ):
        self.preprocessor = preprocessor or (lambda x: x)
        self.postprocessor = postprocessor or (lambda x: x)
        self.uroman = ur.Uroman()
        self.sep = sep
        self.split_alphanumeric = split_alphanumeric
        if backend not in ("python", "rust"):
            raise ValueError(
                f"Invalid backend: {backend}. Choose 'python' or 'rust'."
            )
        self.backend = backend

    def _romanize(self, s: str) -> str:
        if self.backend == "python":
            return uromanize(s)
        else:
            return uromanize_rs(s)

    def _split_alphanumeric_token(self, token: str) -> str:
        """Split tokens with embedded digits like 'di2duan' -> 'di 2 duan'"""
        if not self.split_alphanumeric:
            return token

        # Split on digit boundaries: letter->digit or digit->letter
        result = []
        prev_is_digit = None
        current = []

        for char in token:
            is_digit = char.isdigit()
            if prev_is_digit is not None and prev_is_digit != is_digit:
                # Boundary detected
                result.append("".join(current))
                current = [char]
            else:
                current.append(char)
            prev_is_digit = is_digit

        if current:
            result.append("".join(current))

        return " ".join(result)

    def __call__(self, text: str) -> str:
        text = self.preprocessor(text)
        if self.sep == "":
            tokens = list(text)
        else:
            tokens = text.split(self.sep)
        romanized_units = [self._romanize(token) for token in tokens]

        # Post-process to split alphanumeric tokens
        if self.split_alphanumeric:
            romanized_units = [
                self._split_alphanumeric_token(unit) for unit in romanized_units
            ]

        romanized_text = self.sep.join(romanized_units)
        romanized_text = self.postprocessor(romanized_text)
        return romanized_text


def clean_uromanized_string(s: str) -> str:
    """
    Aggresively cleans a uromanized string.
    """
    s = s.lower()

    # Map quotation marks to a single quote
    PUNCTUATION_TO_NORMALIZE = "'’ʼʻ"
    norm_punct_translation_table = str.maketrans(
        PUNCTUATION_TO_NORMALIZE, "'" * len(PUNCTUATION_TO_NORMALIZE)
    )
    s = s.translate(norm_punct_translation_table)

    # Remove everything else
    s = re.sub("([^a-z' ])", " ", s)

    # Clean single "'"
    s = re.sub(r"[ \n]+'[ \n]+", " ", s)

    # Unify spaces
    s = " ".join(s.split())

    return s.strip()


class UConvInstallationError(Exception): ...


class UConvRomanizationError(Exception): ...


@lru_cache(maxsize=10_000_000)
def uconv_romanize(
    s: str,
    src_script: Optional[str] = None,
    verbose: bool = True,
    uconv_path: str = "uconv",
) -> str:
    """Romanize a string using the uconv tool.

    Args:
        s (str): Input string to romanize.
        src_script (Optional[str], optional): Source script to use for romanization. Defaults to None.

    Raises:
        UConvRomanizationError: If romanization fails.
        UConvRomanizationError: If an unexpected error occurs.

    Returns:
        str: The romanized string.
    """
    if not s.strip():
        return s
    try:
        transliteration_mode = "Any-Latin"
        if src_script and src_script.lower() != "any":
            transliteration_mode = f"{src_script}-Latin;{transliteration_mode}"

        process = subprocess.run(
            [uconv_path, "-x", transliteration_mode],
            input=s,
            text=True,
            capture_output=True,
            check=True,
        )

        if verbose and process.stderr:
            print(f"Warning: {process.stderr}", file=sys.stderr)

        return process.stdout

    except subprocess.SubprocessError as e:
        raise UConvRomanizationError(f"Romanization failed: {e}")
    except Exception as e:
        raise UConvRomanizationError(f"Unexpected error during romanization: {e}")


@lru_cache(maxsize=TROM_CACHE_SIZE)
def _uconv_romanize(self, s: str, src_script: Optional[str] = None) -> str:
    if not s.strip():
        return s
    romanized = uconv_romanize(
        s, src_script=src_script, verbose=self.verbose, uconv_path=self.uconv_path
    )

    return romanized


class UConvRomanizer:
    """
    A class to handle romanization of text using the ICU uconv utility.
    Supports caching of results and custom separation of input text.
    """

    def __init__(
        self,
        preprocessor: Optional[Callable[[str], str]] = None,
        postprocessor: Optional[Callable[[str], str]] = None,
        sep: str = " ",
        uconv_path: str = "uconv",
        verbose: bool = True,
    ):
        """
        Initialize the romanizer with custom separator and uconv path.

        Args:
            sep: Separator to use when splitting/joining text
            uconv_path: Path to uconv executable
            verbose: Whether to print debug information
        """
        self.sep = sep
        self._check_if_installed(uconv_path)
        self.uconv_path = uconv_path
        self.verbose = verbose
        self.preprocessor = preprocessor or (lambda x: x)
        self.postprocessor = postprocessor or (lambda x: x)

    @staticmethod
    def _check_if_installed(uconv_path: str) -> None:
        try:
            subprocess.run(
                [uconv_path, "--version"], check=True, capture_output=True, text=True
            )
        except subprocess.SubprocessError as e:
            raise UConvInstallationError(
                f"uconv is not installed or not accessible: {e}"
            )
        except FileNotFoundError:
            raise UConvInstallationError(f"uconv not found at path: {uconv_path}")

    def _romanize(self, s: str, src_script: Optional[str] = None) -> str:
        return _uconv_romanize(self, s, src_script=src_script)

    def __call__(self, text: str, src_script: Optional[str] = None) -> str:
        if not text.strip():
            return text
        text = self.preprocessor(text)
        if self.sep == "":
            tokens = list(text)
        else:
            tokens = text.split(self.sep)
        romanized = [self._romanize(token, src_script=src_script) for token in tokens]
        romanized_text = self.sep.join(romanized)
        romanized_text = self.postprocessor(romanized_text)
        return romanized_text


class UnifiedRomanizer:
    def __init__(self, sep: str = " ", uconv_path: str = "uconv", verbose: bool = True):
        self.sep = sep
        self.uconv_romanizer = UConvRomanizer(uconv_path=uconv_path, verbose=verbose)
        self.uroman_romanizer = URomanizer()

    def _is_latin(self, char: str) -> bool:
        """Check if a character belongs to the Latin script."""
        try:
            return unicodedata.name(char).startswith(("LATIN", "SPACE", "PUNCTUATION"))
        except ValueError:
            return False

    def _find_unromanized_regions(
        self, original: str, romanized: str
    ) -> list[tuple[int, int]]:
        """
        Find regions in the text that weren't properly romanized by analyzing
        the character properties of the romanized output.
        """
        if not romanized or romanized == original:
            return [(0, len(original))]

        # Check if the romanized text contains only Latin characters and basic punctuation
        if all(self._is_latin(c) for c in romanized):
            return []

        # If we still have non-Latin characters, find those regions
        regions = []
        start = None

        for i, char in enumerate(romanized):
            if not self._is_latin(char):
                if start is None:
                    start = i
            elif start is not None:
                regions.append((start, i))
                start = None

        if start is not None:
            regions.append((start, len(romanized)))

        # Convert regions in romanized text to regions in original text
        original_regions = []
        pos = 0
        for start, end in regions:
            # Find corresponding position in original text
            while pos < len(original) and len(romanized[:start]) > len(original[:pos]):
                pos += 1
            region_start = pos

            while pos < len(original) and len(romanized[:end]) > len(original[:pos]):
                pos += 1
            region_end = pos

            if region_end > region_start:
                original_regions.append((region_start, region_end))

        return original_regions

    def _merge_romanizations(
        self, original: str, first_pass: str, regions: list[tuple[int, int]]
    ) -> str:
        """
        Merge the first-pass romanization with additional romanization of unromanized regions.
        """
        if not regions:
            return first_pass

        result = []
        last_end = 0

        for start, end in regions:
            # Add the successfully romanized portion before this region
            result.append(first_pass[last_end:start])

            # Romanize the unromanized region using uroman
            unromanized_part = original[start:end]
            uroman_result = self.uroman_romanizer(unromanized_part)
            result.append(uroman_result)

            last_end = end

        # Add any remaining romanized portion
        if last_end < len(first_pass):
            result.append(first_pass[last_end:])

        return "".join(result)

    @lru_cache(maxsize=TROM_CACHE_SIZE)
    def _romanize(self, s: str, src_script: Optional[str] = None) -> str:
        """
        Romanize a string using a two-pass approach with improved detection
        of unromanized regions.
        """
        if not s.strip():
            return s

        # First pass: try uconv
        try:
            first_pass = self.uconv_romanizer(s, src_script=src_script)
        except UConvRomanizationError:
            first_pass = s

        # Find regions that weren't properly romanized
        unromanized_regions = self._find_unromanized_regions(s, first_pass)

        # If nothing was romanized or everything needs to be re-romanized, use uroman
        if not unromanized_regions or (
            len(unromanized_regions) == 1 and unromanized_regions[0] == (0, len(s))
        ):
            return self.uroman_romanizer(s)

        # Merge the results of both romanization passes
        romanized = self._merge_romanizations(s, first_pass, unromanized_regions)
        # Last pass go over every character and uromanize every non ascii character (diacritics are allowed)
        final = []
        for char in romanized:
            if not is_latin(char) or char.isspace():
                final.append(self.uroman_romanizer(char))
            else:
                final.append(char)
        return "".join(final)

    def __call__(self, text: str, src_script: Optional[str] = None) -> str:
        tokens = text.split(self.sep)
        romanized = [self._romanize(token, src_script=src_script) for token in tokens]
        return self.sep.join(romanized)


@lru_cache(maxsize=TROM_CACHE_SIZE)
def get_romanizer_for_lang(
    lang: str,
    romanizer_cls: Literal["uroman", "uconv"] = "uroman",
    preprocessor: Optional[
        Callable[[str], str] | Literal["aggressive", "tokenize"]
    ] = None,
    postprocessor: Optional[Callable[[str], str] | Literal["aggressive"]] = None,
    **romanizer_kwargs,
) -> URomanizer | UConvRomanizer:
    """
    Gets a URomanizer or UConvRomanizer instance configured for the specified language.
    This function sets up a romanizer with a language specific pre-processing function.

    Args:
        lang (str): Language code (e.g., 'en', 'de', 'zh').

    Returns:
        URomanizer: Configured URomanizer instance.
    """
    if romanizer_cls not in ("uroman", "uconv"):
        raise ValueError(
            f"Invalid romanizer_cls: {romanizer_cls}. Choose 'uroman' or 'uconv'."
        )

    tokenizer = get_tokenizer_for_lang(lang)

    if preprocessor is None:
        preprocessor = lambda text: text
    elif preprocessor == "aggressive":

        def preprocessor(text: str) -> str:
            if not text.strip():
                return text
            text = text.lower()
            tokens = tokenizer(text)
            text = " ".join(tokens)
            text = " ".join(text.split())  # Normalize spaces
            return text

    elif preprocessor == "tokenize":
        preprocessor = get_spacing_processor(tokenizer)

    if postprocessor is None:
        postprocessor = lambda text: text
    elif postprocessor == "aggressive":
        postprocessor = clean_uromanized_string

    default_lang_kwargs = {
        # Enable more aggressive caching for logographic languages
        "cmn": {"sep": ""},
        "zho": {"sep": ""},
        "jpn": {"sep": ""},
    }
    if lang in default_lang_kwargs:
        for kwarg, val in default_lang_kwargs[lang].items():
            if kwarg not in romanizer_kwargs:
                romanizer_kwargs[kwarg] = val

    if romanizer_cls == "uroman":
        romanizer = URomanizer(
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            **romanizer_kwargs,
        )
    else:
        romanizer = UConvRomanizer(
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            **romanizer_kwargs,
        )

    return romanizer
