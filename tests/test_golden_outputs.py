"""Golden-output regression tests.

The uroman and uroman-rs dependencies are pinned to exact git revisions in
pyproject.toml so that every trom version produces reproducible romanization.
These tests lock in the expected outputs for the pinned revisions: if a rev
bump changes romanization behavior, they fail and the change must be reviewed
(and the golden values updated deliberately).

ICU/uconv output is intentionally not covered here because it depends on the
ICU library version installed on the host, which trom does not pin.
"""

import pytest

from trom import uromanize, uromanize_rs

# (input, lcode, expected romanization)
GOLDEN_SAMPLES = [
    ("Привет мир", None, "Privet mir"),
    ("Γειά σου κόσμε", None, "Geia sou kosme"),
    ("مرحبا بالعالم", None, "mrhba bal'alm"),
    ("שלום עולם", None, "shlvm 'vlm"),
    ("नमस्ते दुनिया", None, "namaste duniyaa"),
    ("你好世界", "cmn", "nihaoshijie"),
    ("こんにちは世界", "jpn", "konnichihashijie"),
    ("안녕하세요 세계", "kor", "annyeonghaseyo segye"),
    ("สวัสดีชาวโลก", "tha", "swatdiichaolok"),
    ("ᎣᏏᏲ ᎡᎶᎯ", None, "osiyo elohi"),
    ("Hello, world! 123", None, "Hello, world! 123"),
]

IDS = [f"{lcode or 'auto'}-{text[:10]}" for text, lcode, _ in GOLDEN_SAMPLES]


@pytest.mark.parametrize("text,lcode,expected", GOLDEN_SAMPLES, ids=IDS)
def test_uroman_python_golden(text, lcode, expected):
    assert uromanize(text, lcode=lcode) == expected


@pytest.mark.parametrize("text,lcode,expected", GOLDEN_SAMPLES, ids=IDS)
def test_uroman_rust_golden(text, lcode, expected):
    assert uromanize_rs(text, lcode=lcode) == expected
