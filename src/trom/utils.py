from functools import lru_cache
from typing import Literal

from iso639 import languages


@lru_cache(maxsize=10_000)
def get_lang_code(
    lang,
    split_on_dash: bool = True,
    what: Literal["part1", "part2b", "part2t", "part3", "name", "full"] = "part3",
) -> str:
    if split_on_dash:
        lang = lang.split("-")[0]

    language = None
    if len(lang) == 3:
        for p in ("part2b", "part2t", "part3"):
            try:
                language = languages.get(**{p: lang})
                break
            except KeyError:
                continue
    elif len(lang) == 2:
        try:
            language = languages.get(part1=lang)
        except KeyError:
            language = None
    elif len(lang) > 3:
        try:
            language = languages.get(name=lang)
        except KeyError:
            language = None

    what = what.lower()
    if what != "full":
        result = getattr(language, what, None)
    else:
        result = language

    if result is None:
        raise ValueError(f"Language code '{lang}' not found in ISO 639-1/2/3.")

    return result
