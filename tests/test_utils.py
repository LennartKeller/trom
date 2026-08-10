"""Tests for utils module."""

import pytest

from trom.utils import get_lang_code


class TestGetLangCode:
    """Test ISO 639 language code resolution."""

    def test_part3_code(self):
        assert get_lang_code("deu") == "deu"

    def test_part2b_code(self):
        """Bibliographic codes must resolve even when they are no part3 code."""
        assert get_lang_code("fre") == "fra"
        assert get_lang_code("ger") == "deu"

    def test_part1_code(self):
        assert get_lang_code("de") == "deu"
        assert get_lang_code("zh") == "zho"

    def test_language_name(self):
        assert get_lang_code("German") == "deu"

    def test_dash_suffix_is_stripped(self):
        assert get_lang_code("de-DE") == "deu"

    def test_unknown_three_letter_code_raises(self):
        with pytest.raises(ValueError, match="not found in ISO 639"):
            get_lang_code("xyz")

    def test_unknown_two_letter_code_raises(self):
        with pytest.raises(ValueError, match="not found in ISO 639"):
            get_lang_code("xy")

    def test_what_parameter(self):
        assert get_lang_code("deu", what="part1") == "de"
        assert get_lang_code("deu", what="name") == "German"
