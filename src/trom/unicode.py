import unicodedata


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
