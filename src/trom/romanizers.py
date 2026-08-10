import logging
import re
import subprocess
from functools import lru_cache
from typing import Callable, Literal, Optional

import uroman as ur
from uroman_rs import Uroman as UromanRS

from .environ import NO_ICU, ROMANIZER_CACHE_SIZE
from .tokenizers import get_tokenizer_for_lang
from .unicode import contains_non_latin_letters, string_is_only_punct

try:
    from icu import Transliterator, UnicodeString

    HAS_PYICU = True
except ImportError:
    HAS_PYICU = False


logger = logging.getLogger(__name__)


class TromRomanizer:
    """
    Base class for all romanizers.

    Provides the shared pre-/postprocessing hooks and separator-based
    tokenization. Subclasses implement `_romanize` for a single unit of text
    and may extend `__call__` with backend-specific keyword arguments.
    """

    def __init__(
        self,
        preprocessor: Optional[Callable[[str], str]] = None,
        postprocessor: Optional[Callable[[str], str]] = None,
        sep: str = " ",
    ):
        self.preprocessor = preprocessor or (lambda x: x)
        self.postprocessor = postprocessor or (lambda x: x)
        self.sep = sep

    def _split(self, text: str) -> list[str]:
        """Split text into units according to the configured separator."""
        if self.sep == "":
            return list(text)
        return text.split(self.sep)

    def _romanize(self, s: str, **kwargs) -> str:
        raise NotImplementedError

    def __call__(self, text: str, **kwargs) -> str:
        text = self.preprocessor(text)
        romanized_units = [self._romanize(unit, **kwargs) for unit in self._split(text)]
        romanized_text = self.sep.join(romanized_units)
        return self.postprocessor(romanized_text)


def get_spacing_processor(
    tokenizer: Callable[[str], list[str]], attempt_reconstruct: bool = False
) -> Callable[[str], str]:
    """Returns a spacing processor that uses the provided tokenizer to add spaces around punctuation.

    Args:
        tokenizer (Callable[[str], list[str]]): Tokenizer function to split the input string into words.

    Returns:
        str: Input string with spaces added around punctuation.
    """
    if attempt_reconstruct:

        def spacing_processor(text: str) -> str:
            words = tokenizer(text)
            if not words:
                return text

            result = []
            for i, token in enumerate(words):
                if string_is_only_punct(token):
                    # Punctuation always attaches to the previous token without a space
                    result.append(token)
                elif i == 0:
                    result.append(token)
                else:
                    result.append(" " + token)
            return "".join(result).strip()
    else:

        def spacing_processor(text: str) -> str:
            words = tokenizer(text)
            return " ".join(words).strip()

    return spacing_processor


def get_spacing_processor_for_lang(lang: str) -> Callable[[str], str]:
    tokenizer = get_tokenizer_for_lang(lang)
    return get_spacing_processor(tokenizer)


@lru_cache(maxsize=ROMANIZER_CACHE_SIZE)
def _get_uroman(*args, **kwargs):
    return ur.Uroman(*args, **kwargs)


@lru_cache(maxsize=ROMANIZER_CACHE_SIZE)
def uromanize(s: str, lcode: Optional[str] = None) -> str:
    return _get_uroman().romanize_string(s, lcode=lcode)


@lru_cache(maxsize=ROMANIZER_CACHE_SIZE)
def _get_uroman_rs(*args, **kwargs):
    return UromanRS(*args, **kwargs)


@lru_cache(maxsize=ROMANIZER_CACHE_SIZE)
def uromanize_rs(s: str, lcode: Optional[str] = None) -> str:
    return _get_uroman_rs().romanize(s, lcode=lcode)


class URomanizer(TromRomanizer):
    def __init__(
        self,
        preprocessor: Optional[Callable[[str], str]] = None,
        postprocessor: Optional[Callable[[str], str]] = None,
        sep: str = " ",
        split_alphanumeric: bool = False,
        backend: Literal["python", "rust"] = "python",
    ):
        super().__init__(
            preprocessor=preprocessor, postprocessor=postprocessor, sep=sep
        )
        self.split_alphanumeric = split_alphanumeric
        if backend not in ("python", "rust"):
            raise ValueError(f"Invalid backend: {backend}. Choose 'python' or 'rust'.")
        self.backend = backend

    def _romanize(self, s: str, lcode: Optional[str] = None) -> str:
        if self.backend == "python":
            return uromanize(s, lcode=lcode)
        else:
            return uromanize_rs(s, lcode=lcode)

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

    def __call__(self, text: str, lcode: Optional[str] = None) -> str:
        text = self.preprocessor(text)
        romanized_units = [
            self._romanize(token, lcode=lcode) for token in self._split(text)
        ]

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
    Aggressively cleans a uromanized string.
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


class ICUTransliterationError(Exception): ...


if HAS_PYICU and not NO_ICU:

    @lru_cache(maxsize=ROMANIZER_CACHE_SIZE)
    def _get_icu_transliterator(src_script: Optional[str] = None) -> Transliterator:
        """
        Create and cache an ICU Transliterator for a given source script.

        `src_script` should be something ICU knows as a transliterator ID prefix,
        e.g. 'Any', 'Greek', 'Cyrillic', 'Katakana', etc.
        """
        if not HAS_PYICU:
            raise ICUTransliterationError(
                "PyICU is not installed. Please `pip install PyICU`."
            )

        # This mirrors your uconv logic:
        #   Any-Latin, optionally prefixed by e.g. Greek-Latin;Any-Latin
        # You can stack more rules here (e.g. Latin-ASCII) if you like.
        translit_id = "Any-Latin"
        if src_script and src_script.lower() != "any":
            translit_id = f"{src_script}-Latin;{translit_id}"

        tr = Transliterator.createInstance(translit_id)
        if tr is None:
            raise ICUTransliterationError(
                f"Could not create Transliterator '{translit_id}'"
            )

        return tr

    @lru_cache(maxsize=ROMANIZER_CACHE_SIZE)
    def icu_romanize(
        s: str,
        src_script: Optional[str] = None,
    ) -> str:
        """
        Romanize a string using ICU Transliterator (PyICU), roughly like `uconv -x`.

        Args:
            s: Input string.
            src_script: Optional source script (e.g. 'Greek', 'Cyrillic', 'Any').

        Returns:
            Romanized string.
        """
        if not s.strip():
            return s

        tr = _get_icu_transliterator(src_script)
        us = UnicodeString(s)
        res = tr.transliterate(us)
        return str(res)


@lru_cache(maxsize=ROMANIZER_CACHE_SIZE)
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
            logger.warning("uconv: %s", process.stderr)

        return process.stdout

    except subprocess.SubprocessError as e:
        raise UConvRomanizationError(f"Romanization failed: {e}")
    except Exception as e:
        raise UConvRomanizationError(f"Unexpected error during romanization: {e}")


class UConvRomanizer(TromRomanizer):
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
        super().__init__(
            preprocessor=preprocessor, postprocessor=postprocessor, sep=sep
        )
        self._check_if_installed(uconv_path)
        self.uconv_path = uconv_path
        self.verbose = verbose

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
        if not s.strip():
            return s
        if HAS_PYICU and not NO_ICU:
            try:
                return icu_romanize(s, src_script=src_script)
            except ICUTransliterationError:
                pass
        return uconv_romanize(
            s, src_script=src_script, verbose=self.verbose, uconv_path=self.uconv_path
        )

    def __call__(self, text: str, src_script: Optional[str] = None) -> str:
        if not text.strip():
            return text
        text = self.preprocessor(text)

        def merciful_romanize(token: str) -> str:
            try:
                return self._romanize(token, src_script=src_script)
            except Exception:
                return token

        romanized = [merciful_romanize(token) for token in self._split(text)]
        romanized_text = self.sep.join(romanized)
        if HAS_PYICU and not NO_ICU:
            romanized_text = re.sub(r" +", " ", romanized_text).strip()
        romanized_text = self.postprocessor(romanized_text)
        return romanized_text


class UnifiedRomanizer(TromRomanizer):
    """
    A unified romanizer that dynamically selects between ICU/uconv and uroman backends.

    This romanizer uses a simple heuristic:
    1. Try ICU/uconv first (preferred for all scripts when available)
    2. If ICU/uconv returns text unchanged (doesn't support the script), fallback to uroman
    3. If uroman also returns unchanged, return original text (merciful mode)

    This approach is much simpler than explicit script detection and works well because:
    - ICU handles most common scripts (Cyrillic, Greek, Arabic, CJK, etc.)
    - uroman fills in gaps for scripts ICU doesn't support (e.g., Cherokee)
    - No complex script detection logic needed

    Args:
        preprocessor: Optional preprocessing function applied before romanization
        postprocessor: Optional postprocessing function applied after romanization
        sep: Separator used for tokenization (default: " ")
        backend_preference: Preferred backend order ("auto", "icu", "uconv", "uroman")
        uroman_backend: Which uroman implementation to use ("python" or "rust")
        uconv_path: Path to uconv executable
        verbose: Whether to print warnings and debug info
        **uroman_kwargs: Additional kwargs passed to URomanizer

    Attributes:
        backends: Dictionary mapping backend names to romanizer instances
        backend_preference: List of backends to try in order
    """

    def __init__(
        self,
        preprocessor: Optional[Callable[[str], str]] = None,
        postprocessor: Optional[Callable[[str], str]] = None,
        sep: str = " ",
        backend_preference: Literal["auto", "icu", "uconv", "uroman"] = "auto",
        uroman_backend: Literal["python", "rust"] = "python",
        uconv_path: str = "uconv",
        verbose: bool = False,
        **uroman_kwargs,
    ):
        super().__init__(
            preprocessor=preprocessor, postprocessor=postprocessor, sep=sep
        )
        self.verbose = verbose

        # Initialize available backends
        self.backends = {}
        self._init_backends(uroman_backend, uconv_path, uroman_kwargs)

        if not self.backends:
            raise RuntimeError(
                "No romanization backends available. Install PyICU, uconv, or uroman."
            )

        # Determine backend preference order
        if backend_preference == "auto":
            # Default order: ICU/uconv > uroman
            self.backend_preference = [
                b for b in ("icu", "uroman") if b in self.backends
            ]
        else:
            # User specified preference. "uconv" and "icu" are served by the same
            # backend (UConvRomanizer prefers in-process ICU and falls back to uconv).
            requested = "icu" if backend_preference == "uconv" else backend_preference
            if requested not in self.backends:
                raise RuntimeError(
                    f"Requested backend '{backend_preference}' is not available. "
                    f"Available backends: {list(self.backends.keys())}"
                )
            self.backend_preference = [requested]

        if self.verbose:
            logger.info(
                "UnifiedRomanizer initialized with backends: %s (preference: %s)",
                list(self.backends.keys()),
                self.backend_preference,
            )

    def _init_backends(
        self,
        uroman_backend: Literal["python", "rust"],
        uconv_path: str,
        uroman_kwargs: dict,
    ):
        """Initialize available romanization backends by reusing existing romanizer classes."""
        # Try to initialize UConvRomanizer (handles both ICU and uconv)
        try:
            uconv_romanizer = UConvRomanizer(
                preprocessor=None,  # We handle pre/post processing at unified level
                postprocessor=None,
                uconv_path=uconv_path,
                verbose=False,  # Suppress UConvRomanizer's own verbose output
            )
            # UConvRomanizer._romanize uses ICU if available, otherwise uconv
            self.backends["icu"] = uconv_romanizer
            if self.verbose:
                if HAS_PYICU and not NO_ICU:
                    logger.info("ICU (PyICU) backend available")
                else:
                    logger.info("uconv backend available at %s", uconv_path)
        except UConvInstallationError:
            if self.verbose:
                logger.info("Neither ICU nor uconv available")

        # Initialize uroman backend (always available as fallback)
        try:
            self.backends["uroman"] = URomanizer(
                preprocessor=None,  # We handle pre/post processing at unified level
                postprocessor=None,
                backend=uroman_backend,
                **uroman_kwargs,
            )
            if self.verbose:
                logger.info(
                    "uroman backend available (%s implementation)", uroman_backend
                )
        except Exception as e:
            if self.verbose:
                logger.info("uroman backend initialization failed: %s", e)

    def _romanize_with_backend(
        self, text: str, backend: str, lcode: Optional[str] = None
    ) -> str:
        """Romanize text using specified backend romanizer."""
        romanizer = self.backends[backend]

        if backend == "icu":
            # UConvRomanizer instance - call its _romanize method
            return romanizer._romanize(text, src_script=None)
        elif backend == "uroman":
            # URomanizer instance - call its _romanize method
            return romanizer._romanize(text, lcode=lcode)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    @staticmethod
    def _has_non_latin_letters(text: str) -> bool:
        """Check if text contains non-Latin letters (excluding punctuation, numbers, spaces)."""
        return contains_non_latin_letters(text)

    def _romanize_text(self, text: str, lcode: Optional[str] = None) -> str:
        """Romanize text using available backends.

        Strategy:
        1. Try ICU/uconv first (preferred)
        2. If ICU/uconv leaves non-Latin letters, fallback to uroman
        3. If uroman also leaves non-Latin letters, return the best available result (merciful mode)
        """
        if not text.strip():
            return text

        icu_result = None

        # Try ICU/uconv backends first (preferred for all scripts)
        for backend in self.backend_preference:
            if backend == "uroman":
                continue  # Skip uroman, save it for fallback
            if backend in self.backends:
                try:
                    result = self._romanize_with_backend(text, backend, lcode=lcode)
                    # Save result for potential fallback
                    icu_result = result

                    # Check if romanization is complete (no non-Latin letters remain)
                    if not self._has_non_latin_letters(result):
                        # Fully romanized, use this result
                        return result
                    else:
                        # ICU partially romanized but non-Latin letters remain
                        if self.verbose:
                            logger.debug(
                                "%s left non-Latin characters, trying uroman", backend
                            )
                        break
                except Exception as e:
                    if self.verbose:
                        logger.debug("%s failed: %s, trying next backend", backend, e)
                    continue

        # Fallback to uroman if ICU/uconv didn't fully romanize
        if "uroman" in self.backends:
            try:
                result = self._romanize_with_backend(text, "uroman", lcode=lcode)
                # Use uroman result if it's better (more romanized)
                if not self._has_non_latin_letters(result):
                    return result
                # If uroman also didn't fully romanize, prefer uroman result anyway
                # (it might handle scripts ICU missed)
                if icu_result is None or self._has_non_latin_letters(icu_result):
                    return result
            except Exception as e:
                if self.verbose:
                    logger.debug("uroman fallback failed: %s", e)

        # Return ICU result if we have it, otherwise return original (merciful mode)
        return icu_result if icu_result is not None else text

    def __call__(self, text: str, lcode: Optional[str] = None) -> str:
        """
        Romanize text using ICU/uconv first, with uroman as fallback.

        Args:
            text: Input text to romanize (may contain mixed scripts)
            lcode: Optional language code (used for CJK languages with uroman)

        Returns:
            Romanized text
        """
        if not text.strip():
            return text

        # Apply preprocessing
        text = self.preprocessor(text)

        # Handle separator-based tokenization
        if self.sep == "":
            # Character-by-character mode - romanize entire text at once
            romanized_text = self._romanize_text(text, lcode=lcode)
        else:
            # Token-based mode with separator
            tokens = text.split(self.sep)
            romanized_tokens = []

            for token in tokens:
                if not token.strip():
                    romanized_tokens.append(token)
                    continue

                # Romanize each token
                romanized = self._romanize_text(token, lcode=lcode)
                romanized_tokens.append(romanized)

            romanized_text = self.sep.join(romanized_tokens)

        # Clean up multiple spaces if using ICU/uconv
        if "icu" in self.backends or "uconv" in self.backends:
            romanized_text = re.sub(r" +", " ", romanized_text).strip()

        # Apply postprocessing
        romanized_text = self.postprocessor(romanized_text)

        return romanized_text


@lru_cache(maxsize=ROMANIZER_CACHE_SIZE)
def get_romanizer_for_lang(
    lang: str,
    romanizer_cls: Literal["uroman", "uconv", "unified"] = "uroman",
    preprocessor: Optional[
        Callable[[str], str] | Literal["aggressive", "tokenize"]
    ] = None,
    postprocessor: Optional[Callable[[str], str] | Literal["aggressive"]] = None,
    **romanizer_kwargs,
) -> URomanizer | UConvRomanizer | UnifiedRomanizer:
    """
    Gets a URomanizer, UConvRomanizer, or UnifiedRomanizer instance configured for the specified language.
    This function sets up a romanizer with a language specific pre-processing function.

    Args:
        lang (str): Language code (e.g., 'en', 'de', 'zh').
        romanizer_cls (str): Romanizer class to use ('uroman', 'uconv', or 'unified').
        preprocessor: Optional preprocessing function or preset ('aggressive', 'tokenize').
        postprocessor: Optional postprocessing function or preset ('aggressive').
        **romanizer_kwargs: Additional kwargs passed to the romanizer constructor.

    Returns:
        URomanizer | UConvRomanizer | UnifiedRomanizer: Configured romanizer instance.
    """
    if romanizer_cls not in ("uroman", "uconv", "unified"):
        raise ValueError(
            f"Invalid romanizer_cls: {romanizer_cls}. Choose 'uroman', 'uconv', or 'unified'."
        )

    tokenizer = get_tokenizer_for_lang(lang)

    if preprocessor is None:

        def preprocessor(text: str) -> str:
            return text

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

        def postprocessor(text: str) -> str:
            return text

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
    elif romanizer_cls == "uconv":
        romanizer = UConvRomanizer(
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            **romanizer_kwargs,
        )
    else:  # unified
        romanizer = UnifiedRomanizer(
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            **romanizer_kwargs,
        )

    return romanizer
