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

    if len(lang) == 3:
        for p in ("part2b", "part2t", "part3"):
            l = None
            try:
                l = languages.get(**{p: lang})
            except KeyError:
                continue
    elif len(lang) == 2:
        l = languages.get(part1=lang)
    elif len(lang) > 3:
        l = languages.get(name=lang)
    else:
        l = None

    what = what.lower()
    if what != "full":
        result = getattr(l, what, None)
    else:
        result = l

    if result is None:
        raise ValueError(f"Language code '{lang}' not found in ISO 639-1/2/3.")

    return result
