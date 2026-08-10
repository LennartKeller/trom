"""Command line interface for trom."""

import sys
from typing import Optional

import click

from .romanizers import get_romanizer_for_lang
from .utils import get_lang_code


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("text", nargs=-1)
@click.option(
    "-l",
    "--lang",
    default="eng",
    show_default=True,
    help="ISO 639 language code of the input (e.g. eng, cmn, jpn, rus).",
)
@click.option(
    "-r",
    "--romanizer",
    "romanizer_cls",
    type=click.Choice(["uroman", "uconv", "unified"]),
    default="unified",
    show_default=True,
    help="Romanizer implementation to use.",
)
@click.option(
    "-b",
    "--backend",
    type=click.Choice(["python", "rust"]),
    default="python",
    show_default=True,
    help="uroman backend (ignored for the uconv romanizer).",
)
@click.option(
    "--pre",
    "preprocessor",
    type=click.Choice(["aggressive", "tokenize"]),
    default=None,
    help="Preprocessing preset applied before romanization.",
)
@click.option(
    "--post",
    "postprocessor",
    type=click.Choice(["aggressive"]),
    default=None,
    help="Postprocessing preset applied after romanization.",
)
@click.option(
    "-i",
    "--input",
    "input_file",
    type=click.File("r", encoding="utf-8"),
    default=None,
    help="Read input from a file instead of arguments/stdin.",
)
@click.option(
    "-o",
    "--output",
    type=click.File("w", encoding="utf-8"),
    default="-",
    show_default=True,
    help="Write output to a file (default: stdout).",
)
def main(
    text: tuple[str, ...],
    lang: str,
    romanizer_cls: str,
    backend: str,
    preprocessor: Optional[str],
    postprocessor: Optional[str],
    input_file,
    output,
) -> None:
    """Romanize TEXT.

    Reads from TEXT arguments, --input, or stdin (in that order of precedence)
    and writes romanized text line by line.

    \b
    Examples:
        trom -l rus "Привет мир"
        trom -l cmn -r uroman "你好世界"
        cat corpus.txt | trom -l ell > corpus.rom.txt
    """
    try:
        lcode = get_lang_code(lang)
    except ValueError as e:
        raise click.BadParameter(str(e), param_hint="--lang")

    romanizer_kwargs = {}
    if romanizer_cls in ("uroman", "unified"):
        key = "backend" if romanizer_cls == "uroman" else "uroman_backend"
        romanizer_kwargs[key] = backend

    romanizer = get_romanizer_for_lang(
        lang,
        romanizer_cls=romanizer_cls,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        **romanizer_kwargs,
    )

    if text:
        lines = [" ".join(text)]
    elif input_file is not None:
        lines = input_file
    else:
        if sys.stdin.isatty():
            raise click.UsageError(
                "No input given. Pass TEXT arguments, --input, or pipe to stdin."
            )
        lines = sys.stdin

    for line in lines:
        line = line.rstrip("\n")
        if romanizer_cls == "uconv":
            romanized = romanizer(line)
        else:
            romanized = romanizer(line, lcode=lcode)
        output.write(romanized + "\n")


if __name__ == "__main__":
    main()
