from functools import lru_cache
from typing import Literal

from iso639 import Lang
from iso639.exceptions import InvalidLanguageValue

_FIELDS = {
    "part1": "pt1",
    "part2b": "pt2b",
    "part2t": "pt2t",
    "part3": "pt3",
    "name": "name",
}


@lru_cache(maxsize=10_000)
def get_lang_code(
    lang,
    split_on_dash: bool = True,
    what: Literal["part1", "part2b", "part2t", "part3", "name", "full"] = "part3",
) -> str:
    if split_on_dash:
        lang = lang.split("-")[0]

    try:
        language = Lang(lang)
    except InvalidLanguageValue:
        raise ValueError(f"Language code '{lang}' not found in ISO 639-1/2/3.")

    what = what.lower()
    if what == "full":
        return language

    result = getattr(language, _FIELDS[what], None)
    if not result:  # iso639-lang uses "" for missing codes
        raise ValueError(f"Language code '{lang}' has no {what} code.")

    return result
