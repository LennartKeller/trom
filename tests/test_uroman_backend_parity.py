"""Backend-parity tests between the Python uroman and Rust uroman-rs backends.

uroman_backend_parity.tsv holds sentences sampled from FineWeb2 covering all
non-Latin scripts in the corpus. For every sentence, uromanize_rs must produce
byte-identical output to the original Python uroman implementation.

Known divergences between the pinned backend revisions are listed in
KNOWN_DIVERGENT and marked xfail(strict=True): if a rev bump brings the
backends into agreement, the XPASS fails the suite and the entry must be
removed.
"""

import csv
from pathlib import Path

import pytest

from trom import uromanize, uromanize_rs

TSV_PATH = Path(__file__).parent / "uroman_backend_parity.tsv"

# (lcode, exact input text) of rows where the pinned backends disagree:
# - pan: Python uroman inconsistently keeps a trailing inherent vowel after
#   word-final geminates ("ajja", "pabba") where Rust drops it ("ajj", "pabb");
#   this is a bug in the Python reference, not in uroman-rs.
KNOWN_DIVERGENT = {
    ("pan", "ਉਨ੍ਹਾਂ ਨੇ ਅੱਜ ਦੀ ਦਿਹਾੜੀ ਦਾ ਕੰਮ ਖ਼ਤਮ ਕੀਤਾ ਤਾਂ ਸਿੱਧੇ ਪੱਬ ਵਿਚ ਜਾ ਬੜੇ।"),
    ("pan", "ਬਿਲਡਿੰਗ ਕੌਨਟਰੈਕਟਰ ਦੀ ਵੈਨ ਖ਼ਰਾਬ ਸੀ ਤੇ ਉਸ ਦੀ ਤਬੀਅਤ ਵੀ ਨਾਸਾਜ਼ ਸੀ।"),
}


def _load_rows():
    with TSV_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


ROWS = _load_rows()

PARAMS = [
    pytest.param(
        row,
        marks=pytest.mark.xfail(
            (row["lcode"], row["text"]) in KNOWN_DIVERGENT,
            reason="known divergence between pinned uroman and uroman-rs revisions",
            strict=True,
        ),
        id=f"{row['lcode']}-{row['script']}-{i}",
    )
    for i, row in enumerate(ROWS)
]


@pytest.mark.parametrize("row", PARAMS)
def test_rust_backend_matches_python_uroman(row):
    lcode = row["lcode"] if row["lcode"] != "und" else None
    assert uromanize_rs(row["text"], lcode=lcode) == uromanize(row["text"], lcode=lcode)
