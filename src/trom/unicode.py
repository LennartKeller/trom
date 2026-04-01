import re
import unicodedata
from typing import List


def nfd(s: str) -> str:
    return unicodedata.normalize("NFD", s)


def nfkd(s: str) -> str:
    return unicodedata.normalize("NFKD", s)


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def print_unicode_chars(s: str) -> None:
    for c in s:
        print(c, ord(c), repr(c))


def char_is_punct(char: str) -> bool:
    """Detects all punctuation characters in Unicode."""
    return unicodedata.category(char).startswith("P")


def string_is_only_punct(s: str) -> bool:
    """
    Checks if a string is only made up of punctuation characters or whitespaces.
    """
    return all(char_is_punct(char) or char.isspace() for char in s)


def string_contains_punct(s: str) -> bool:
    return any(char_is_punct(char) for char in s)


def get_puntuation(s: str) -> str:
    """Returns all punctuation characters in a string."""
    return "".join(char for char in s if char_is_punct(char))


def is_latin(char):
    """
    Check if a character is Latin (including characters with diacritical marks).

    Args:
        char (str): A single character to check.

    Returns:
        bool: True if the character is Latin, False otherwise.

    Raises:
        ValueError: If input is not a single character.
    """
    if len(char) != 1:
        return all(is_latin(c) for c in char)
    category = unicodedata.category(char)
    if category.startswith("L"):
        try:
            name = unicodedata.name(char)
            return "LATIN" in name
        except ValueError:
            return False
    return False


def is_latin_only(text):
    for char in text:
        # Get the formal Unicode name (e.g., 'LATIN SMALL LETTER A')
        try:
            name = unicodedata.name(char)
        except ValueError:
            # Characters with no name (like some control codes) are usually not Latin
            continue

        # We allow numbers, punctuation, and symbols regardless of "script"
        # Categories: 'P' (Punctuation), 'N' (Number), 'Z' (Separator/Space), 'S' (Symbol)
        category = unicodedata.category(char)
        if category[0] in ("P", "N", "Z", "S"):
            continue

        # Check if the character is a Letter ('L') and specifically 'LATIN'
        if category.startswith("L"):
            if "LATIN" not in name:
                return False

    return True


def utf8len(s: str) -> int:
    return len(s.encode("utf-8"))


def get_script(char: str) -> str:
    """
    Detect the Unicode script of a character.

    Args:
        char: A single character to detect script for.

    Returns:
        str: Script name (e.g., 'Latin', 'Cyrillic', 'Han', 'Arabic', etc.)
             Returns 'Common' for punctuation and symbols.
             Returns 'Unknown' if script cannot be determined.
    """
    if len(char) != 1:
        raise ValueError("get_script() expects a single character")

    # Handle whitespace and basic ASCII
    if char.isspace():
        return "Common"

    code_point = ord(char)

    # Common punctuation and symbols
    if char_is_punct(char) or unicodedata.category(char) in ("Sm", "Sc", "Sk", "So"):
        return "Common"

    # Digits
    if char.isdigit():
        return "Common"

    # Use Unicode name to detect script for letters
    try:
        name = unicodedata.name(char)

        # Common scripts
        if "LATIN" in name:
            return "Latin"
        elif "CYRILLIC" in name:
            return "Cyrillic"
        elif "GREEK" in name:
            return "Greek"
        elif "ARABIC" in name:
            return "Arabic"
        elif "HEBREW" in name:
            return "Hebrew"
        elif "DEVANAGARI" in name:
            return "Devanagari"
        elif "THAI" in name:
            return "Thai"
        elif "HANGUL" in name or "JONGSEONG" in name or "CHOSEONG" in name:
            return "Hangul"
        elif "HIRAGANA" in name:
            return "Hiragana"
        elif "KATAKANA" in name:
            return "Katakana"
        elif "CJK" in name or "IDEOGRAPH" in name:
            return "Han"
        elif "KHMER" in name:
            return "Khmer"
        elif "LAO" in name:
            return "Lao"
        elif "MYANMAR" in name:
            return "Myanmar"
        elif "TIBETAN" in name:
            return "Tibetan"
        elif "GEORGIAN" in name:
            return "Georgian"
        elif "ARMENIAN" in name:
            return "Armenian"
        elif "ETHIOPIC" in name:
            return "Ethiopic"
        elif "CHEROKEE" in name:
            return "Cherokee"

    except ValueError:
        pass

    # Fallback: use code point ranges for CJK and other scripts
    # CJK Unified Ideographs
    if 0x4E00 <= code_point <= 0x9FFF:
        return "Han"
    # CJK Extension A
    elif 0x3400 <= code_point <= 0x4DBF:
        return "Han"
    # Hiragana
    elif 0x3040 <= code_point <= 0x309F:
        return "Hiragana"
    # Katakana
    elif 0x30A0 <= code_point <= 0x30FF:
        return "Katakana"
    # Hangul Syllables
    elif 0xAC00 <= code_point <= 0xD7AF:
        return "Hangul"
    # Cherokee
    elif 0x13A0 <= code_point <= 0x13FF:
        return "Cherokee"

    return "Unknown"


def detect_script_spans(text: str) -> List[tuple[str, int, int]]:
    """
    Detect spans of different scripts in text.

    Args:
        text: Input text to analyze.

    Returns:
        List of tuples (script, start_idx, end_idx) where each tuple represents
        a continuous span of characters belonging to the same script.
        'Common' script characters are merged with adjacent non-Common scripts.
    """
    if not text:
        return []

    spans = []
    current_script = None
    start_idx = 0

    for i, char in enumerate(text):
        script = get_script(char)

        if current_script is None:
            current_script = script
            start_idx = i
        elif script != current_script and script != "Common":
            # End current span and start new one
            # But first, extend previous span if it was Common
            if current_script == "Common" and spans:
                # Merge with previous span
                prev_script, prev_start, _ = spans[-1]
                spans[-1] = (prev_script, prev_start, i)
            else:
                spans.append((current_script, start_idx, i))
            current_script = script
            start_idx = i
        # If new char is Common, we don't change current_script
        # (Common chars are attached to their context)

    # Add final span
    if current_script is not None:
        if current_script == "Common" and spans:
            # Merge with previous span
            prev_script, prev_start, _ = spans[-1]
            spans[-1] = (prev_script, prev_start, len(text))
        else:
            spans.append((current_script, start_idx, len(text)))

    return spans


def get_unicode_codepoints(s: str, as_hex: bool = True):
    """
    Return the list of Unicode code points for a given UTF string.

    Args:
        s (str): Input string.
        as_hex (bool): If True, return code points in hexadecimal (e.g. 'U+1F600'),
                       otherwise return as integers.

    Returns:
        list[str] | list[int]: Unicode code points of all characters.
    """
    if as_hex:
        return [f"U+{ord(c):04X}" for c in s]
    else:
        return [ord(c) for c in s]


def split_utf8_near_whitespace(text: str, max_bytes: int = 49149 - 100) -> List[str]:
    """
    Split `text` into segments of at most `max_bytes` UTF-8 bytes.
    Prefer splits at whitespace; if a single token exceeds `max_bytes`,
    hard-split it without breaking multibyte characters.

    Returns a list of non-empty segments (whitespace at edges is stripped).
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")

    # Tokens: either runs of whitespace or runs of non-whitespace
    tokens = re.findall(r"\s+|\S+", text)

    parts: List[str] = []
    current: List[str] = []
    cur_bytes = 0

    def hard_split_token(tok: str) -> List[str]:
        """Split a single overlong token into <= max_bytes chunks safely."""
        chunks, buf, buf_bytes = [], [], 0
        for ch in tok:
            b = utf8len(ch)
            if b > max_bytes:
                # A single character exceeds the limit — extremely rare for UTF-8,
                # but handle defensively by placing it alone.
                if buf:
                    chunks.append("".join(buf))
                    buf, buf_bytes = [], 0
                chunks.append(ch)
                continue
            if buf_bytes + b > max_bytes:
                chunks.append("".join(buf))
                buf, buf_bytes = [ch], b
            else:
                buf.append(ch)
                buf_bytes += b
        if buf:
            chunks.append("".join(buf))
        return chunks

    for tok in tokens:
        tok_b = utf8len(tok)

        if tok_b > max_bytes:
            # Flush current before hard-splitting the huge token
            if current:
                parts.append("".join(current).strip())
                current, cur_bytes = [], 0
            parts.extend(hard_split_token(tok))
            continue

        if cur_bytes + tok_b <= max_bytes:
            current.append(tok)
            cur_bytes += tok_b
        else:
            # Close at the nearest earlier whitespace boundary (thanks to tokenization)
            if current:
                parts.append("".join(current).strip())
            current = [tok]
            cur_bytes = tok_b

    if current:
        parts.append("".join(current).strip())

    # Remove any empty strings (could occur if text is all whitespace)
    assert all(utf8len(p) <= max_bytes for p in parts)
    return [p for p in parts if p]


EXTRA_MAP = str.maketrans(
    {
        "ß": "ss",
        "ẞ": "SS",
        "ł": "l",
        "Ł": "L",
        "ø": "o",
        "Ø": "O",
        "đ": "d",
        "Đ": "D",
        "ð": "d",
        "Ð": "D",
        "þ": "th",
        "Þ": "Th",
        "æ": "ae",
        "Æ": "AE",
        "œ": "oe",
        "Œ": "OE",
    }
)


def remove_diacritics(text: str) -> str:
    text = text.translate(EXTRA_MAP)
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
