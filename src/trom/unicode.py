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


def utf8len(s: str) -> int:
    return len(s.encode("utf-8"))


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
