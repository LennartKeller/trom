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
