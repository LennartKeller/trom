import logging
import re
import subprocess
from functools import lru_cache
from typing import Callable, Literal, Optional

import uroman as ur
from uroman_rs import Uroman as UromanRS

from .environ import NO_ICU, ROMANIZER_CACHE_SIZE
from .tokenizers import get_tokenizer_for_lang
from .unicode import (
    RESIDUAL_RUN_RE,
    compile_script_run_re,
    remove_diacritics,
    string_is_only_punct,
)

try:
    from icu import Transliterator, UnicodeString

    HAS_PYICU = True
    PYICU_IMPORT_ERROR = None
except ImportError as e:
    HAS_PYICU = False
    # Kept for diagnostics: distinguishes "PyICU not installed" from a build
    # linked against an ICU that no longer exists on this machine.
    PYICU_IMPORT_ERROR = str(e)


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


def clean_romanized_string(s: str) -> str:
    """
    Aggressively clean a romanized string down to lowercase ASCII, tone-safely.

    Counterpart to `clean_uromanized_string` for output that may come from the
    ICU pass of `UnifiedRomanizer`: diacritics are folded into their base
    letters (rather than stripped into spaces) and digits are kept, so tone
    information encoded as numeric pinyin survives. To get tone digits into
    the text in the first place, romanize with
    `UnifiedRomanizer(numeric_pinyin=True)`; without that, ICU marks tones
    with diacritics and folding removes them.
    """
    s = s.lower()

    # Map quotation marks to a single quote
    PUNCTUATION_TO_NORMALIZE = "'’ʼʻ"
    norm_punct_translation_table = str.maketrans(
        PUNCTUATION_TO_NORMALIZE, "'" * len(PUNCTUATION_TO_NORMALIZE)
    )
    s = s.translate(norm_punct_translation_table)

    # Fold diacritics into base letters (ǔ -> u) instead of deleting the
    # whole character like clean_uromanized_string does
    s = remove_diacritics(s)

    # Remove everything else, keeping digits (numeric pinyin tones)
    s = re.sub("([^a-z0-9' ])", " ", s)

    # Clean single "'"
    s = re.sub(r"[ \n]+'[ \n]+", " ", s)

    # Unify spaces
    s = " ".join(s.split())

    return s.strip()


class UConvInstallationError(Exception): ...


class UConvRomanizationError(Exception): ...


class ICUTransliterationError(Exception): ...


def _build_translit_id(src_script: Optional[str] = None) -> str:
    # Any-Latin, optionally prefixed by e.g. Greek-Latin;Any-Latin
    translit_id = "Any-Latin"
    if src_script and src_script.lower() != "any":
        translit_id = f"{src_script}-Latin;{translit_id}"
    return translit_id


if HAS_PYICU and not NO_ICU:

    @lru_cache(maxsize=ROMANIZER_CACHE_SIZE)
    def _get_icu_transliterator(translit_id: str) -> Transliterator:
        """Create and cache an ICU Transliterator for a full transform ID."""
        tr = Transliterator.createInstance(translit_id)
        if tr is None:
            raise ICUTransliterationError(
                f"Could not create Transliterator '{translit_id}'"
            )

        return tr

    @lru_cache(maxsize=ROMANIZER_CACHE_SIZE)
    def icu_transliterate(s: str, translit_id: str) -> str:
        """Apply an arbitrary ICU transform (e.g. 'Han-Latin; Latin-NumericPinyin')."""
        if not s.strip():
            return s

        tr = _get_icu_transliterator(translit_id)
        us = UnicodeString(s)
        res = tr.transliterate(us)
        return str(res)

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
        return icu_transliterate(s, _build_translit_id(src_script))


@lru_cache(maxsize=ROMANIZER_CACHE_SIZE)
def uconv_transliterate(
    s: str,
    translit_id: str,
    verbose: bool = True,
    uconv_path: str = "uconv",
) -> str:
    """Apply an arbitrary ICU transform via the uconv tool."""
    if not s.strip():
        return s
    try:
        process = subprocess.run(
            [uconv_path, "-x", translit_id],
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
    return uconv_transliterate(
        s, _build_translit_id(src_script), verbose=verbose, uconv_path=uconv_path
    )


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
    A unified romanizer combining ICU/uconv fidelity with uroman coverage.

    Romanization happens in two passes at *residual-run* granularity:

    1. ICU/uconv Any-Latin transliterates the whole text. ICU is inherently
       robust to mixed scripts: it romanizes the characters it supports and
       passes the rest through untouched.
    2. Any maximal run of characters still outside the Latin, Common, and
       Inherited scripts is romanized with uroman. Whatever neither backend
       handles passes through unchanged (merciful mode).

    Compared to deciding per token whether ICU output "looks romanized", this
    keeps the backend choice a per-character-run property of the input: one
    stray character ICU doesn't know (common in noisy web text) costs uroman
    on that run only, not the ICU fidelity of the surrounding document, and
    ICU transliteration artifacts (modifier letters, combining marks) can
    never trigger a spurious fallback.

    Args:
        preprocessor: Optional preprocessing function applied before romanization
        postprocessor: Optional postprocessing function applied after romanization
        sep: Accepted for API compatibility; romanization is whole-text and
            no longer splits on it
        backend_preference: "auto" (ICU first, uroman for residual runs),
            "icu"/"uconv" (ICU only, residual runs pass through), or
            "uroman" (uroman only)
        uroman_backend: Which uroman implementation to use ("python" or "rust")
        uconv_path: Path to uconv executable
        verbose: Whether to print warnings and debug info
        script_overrides: Optional mapping of Unicode script names to how their
            runs are romanized instead of the ICU pass: the string "uroman", or
            an ICU transform ID (e.g. {"Thai": "uroman",
            "Han": "Han-Latin; Latin-NumericPinyin"}). Runs are carved out
            before pass 1, so this composes with the residual-run strategy.
        numeric_pinyin: Convenience switch adding
            {"Han": "Han-Latin; Latin-NumericPinyin"} to script_overrides, so
            Mandarin tones survive later ASCII folding as digits
            (see clean_romanized_string)
        **uroman_kwargs: Additional kwargs passed to URomanizer

    Attributes:
        backends: Dictionary mapping backend names to romanizer instances
        backend_preference: List of backends in use, in pass order
    """

    _emitted_warnings: set[str] = set()

    @classmethod
    def _warn_once(cls, message: str) -> None:
        if message not in cls._emitted_warnings:
            cls._emitted_warnings.add(message)
            logger.warning(message)

    def __init__(
        self,
        preprocessor: Optional[Callable[[str], str]] = None,
        postprocessor: Optional[Callable[[str], str]] = None,
        sep: str = " ",
        backend_preference: Literal["auto", "icu", "uconv", "uroman"] = "auto",
        uroman_backend: Literal["python", "rust"] = "python",
        uconv_path: str = "uconv",
        verbose: bool = False,
        script_overrides: Optional[dict[str, str]] = None,
        numeric_pinyin: bool = False,
        **uroman_kwargs,
    ):
        super().__init__(
            preprocessor=preprocessor, postprocessor=postprocessor, sep=sep
        )
        self.verbose = verbose
        self.uconv_path = uconv_path

        # Initialize available backends
        self.backends = {}
        self._init_backends(uroman_backend, uconv_path, uroman_kwargs)

        if not self.backends:
            raise RuntimeError(
                "No romanization backends available. Install PyICU, uconv, or uroman."
            )

        # Determine which passes run. "uconv" and "icu" are served by the same
        # backend (UConvRomanizer prefers in-process ICU and falls back to uconv).
        if backend_preference == "auto":
            self.backend_preference = [
                b for b in ("icu", "uroman") if b in self.backends
            ]
        else:
            requested = "icu" if backend_preference == "uconv" else backend_preference
            if requested not in self.backends:
                raise RuntimeError(
                    f"Requested backend '{backend_preference}' is not available. "
                    f"Available backends: {list(self.backends.keys())}"
                )
            self.backend_preference = [requested]
        self._use_icu = "icu" in self.backend_preference
        self._use_uroman = "uroman" in self.backend_preference

        # Per-script overrides, carved out of the text before the ICU pass.
        self.script_overrides = dict(script_overrides or {})
        if numeric_pinyin:
            self.script_overrides.setdefault("Han", "Han-Latin; Latin-NumericPinyin")
        self._script_run_res = {}
        for script, target in self.script_overrides.items():
            needed = "uroman" if target == "uroman" else "icu"
            if needed not in self.backends or needed not in self.backend_preference:
                logger.warning(
                    "Ignoring script override %s -> %r: %s backend not in use",
                    script,
                    target,
                    needed,
                )
                continue
            # Raises ValueError for unknown script names
            self._script_run_res[script] = compile_script_run_re(script)

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
            if not HAS_PYICU and not NO_ICU:
                # Same transforms, but the subprocess is slower and the host's
                # ICU version is not pinned by trom -- outputs may differ
                # between machines. (A deliberate NO_ICU opt-out stays quiet.)
                error = PYICU_IMPORT_ERROR or ""
                if len(error) > 150:
                    error = error[:150] + "..."
                self._warn_once(
                    f"PyICU failed to import ({error}); on macOS, run "
                    "install_icupy_macos.sh to rebuild it. Falling back to the "
                    f"uconv subprocess at {uconv_path} (unpinned ICU version, slower)"
                )
        except UConvInstallationError:
            self._warn_once(
                "Neither PyICU nor uconv is available; UnifiedRomanizer will "
                "romanize with uroman only"
            )

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

    def _icu_transform(self, s: str, translit_id: str) -> str:
        if HAS_PYICU and not NO_ICU:
            return icu_transliterate(s, translit_id)
        return uconv_transliterate(
            s, translit_id, verbose=self.verbose, uconv_path=self.uconv_path
        )

    def _uroman_run(self, run: str, lcode: Optional[str] = None) -> str:
        try:
            return self.backends["uroman"]._romanize(run, lcode=lcode)
        except Exception as e:
            if self.verbose:
                logger.debug("uroman failed on %r: %s", run, e)
            return run

    def _patch_residual_runs(self, text: str, lcode: Optional[str] = None) -> str:
        """Pass 2: romanize with uroman every run ICU left in a foreign script."""
        pieces = []
        pos = 0
        for m in RESIDUAL_RUN_RE.finditer(text):
            pieces.append(text[pos : m.start()])
            pieces.append(self._uroman_run(m.group(), lcode=lcode))
            pos = m.end()
        if pos == 0:
            return text
        pieces.append(text[pos:])
        return "".join(pieces)

    def _romanize_segment(self, segment: str, lcode: Optional[str] = None) -> str:
        """Run the ICU pass and the uroman residual pass over one text segment."""
        result = segment
        if self._use_icu:
            try:
                result = self.backends["icu"]._romanize(segment, src_script=None)
            except Exception as e:
                if self.verbose:
                    logger.debug("ICU pass failed: %s", e)
                result = segment
        if self._use_uroman:
            result = self._patch_residual_runs(result, lcode=lcode)
        return result

    def _iter_override_runs(self, text: str):
        """Yield (start, end, script, run) for script-override runs, in order."""
        matches = []
        for script, run_re in self._script_run_res.items():
            for m in run_re.finditer(text):
                matches.append((m.start(), m.end(), script, m.group()))
        matches.sort()
        return matches

    def _romanize_override_run(
        self, run: str, script: str, lcode: Optional[str] = None
    ) -> str:
        target = self.script_overrides[script]
        if target == "uroman":
            return self._uroman_run(run, lcode=lcode)
        try:
            result = self._icu_transform(run, target)
        except Exception as e:
            if self.verbose:
                logger.debug("override transform %r failed: %s", target, e)
            result = run
        # Characters the transform has no mapping for (e.g. rare Han) still
        # get the uroman fallback.
        if self._use_uroman:
            result = self._patch_residual_runs(result, lcode=lcode)
        return result

    def _romanize_text(self, text: str, lcode: Optional[str] = None) -> str:
        if not text.strip():
            return text

        if not self._script_run_res:
            return self._romanize_segment(text, lcode=lcode)

        pieces = []
        pos = 0
        for start, end, script, run in self._iter_override_runs(text):
            if start < pos:  # overlap safety; should not happen for distinct scripts
                continue
            if start > pos:
                pieces.append(self._romanize_segment(text[pos:start], lcode=lcode))
            pieces.append(self._romanize_override_run(run, script, lcode=lcode))
            pos = end
        if pos < len(text):
            pieces.append(self._romanize_segment(text[pos:], lcode=lcode))
        return "".join(pieces)

    def __call__(self, text: str, lcode: Optional[str] = None) -> str:
        """
        Romanize text: ICU pass first, uroman on residual non-Latin runs.

        Args:
            text: Input text to romanize (may contain mixed scripts)
            lcode: Optional language code (passed to uroman)

        Returns:
            Romanized text
        """
        if not text.strip():
            return text

        text = self.preprocessor(text)
        romanized_text = self._romanize_text(text, lcode=lcode)

        # Clean up multiple spaces if the ICU pass is active
        if self._use_icu:
            romanized_text = re.sub(r" +", " ", romanized_text).strip()

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
