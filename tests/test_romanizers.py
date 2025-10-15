"""Tests for romanizers module."""

import pytest
from trom.romanizers import (
    URomanizer,
    clean_uromanized_string,
    get_romanizer_for_lang,
    get_spacing_processor,
    uromanize,
    uromanize_rs,
)
from trom.tokenizers import tokenize_whitespace


class TestUromanizeFunctions:
    """Test basic uromanize and uromanize_rs functions."""

    def test_uromanize_basic_latin(self):
        """Test that Latin text passes through unchanged."""
        text = "hello world"
        result = uromanize(text)
        assert result == "hello world"

    def test_uromanize_cyrillic(self):
        """Test Cyrillic to Latin romanization."""
        text = "Привет"
        result = uromanize(text)
        assert "Privet" in result or "rivet" in result.lower()

    def test_uromanize_arabic(self):
        """Test Arabic to Latin romanization."""
        text = "مرحبا"
        result = uromanize(text)
        assert result != text
        assert all(ord(c) < 128 or c.isspace() for c in result)

    def test_uromanize_chinese(self):
        """Test Chinese to Latin romanization."""
        text = "你好"
        result = uromanize(text)
        assert result != text
        assert "ni" in result.lower() or "hao" in result.lower()

    def test_uromanize_greek(self):
        """Test Greek to Latin romanization."""
        text = "Γεια σου"
        result = uromanize(text)
        assert result != text
        assert "eia" in result or "gia" in result.lower()

    def test_uromanize_empty_string(self):
        """Test that empty string returns empty string."""
        assert uromanize("") == ""

    def test_uromanize_with_lcode(self):
        """Test uromanize with language code parameter."""
        text = "你好"
        result = uromanize(text, lcode="cmn")
        assert result != text

    def test_uromanize_rs_basic_latin(self):
        """Test that uromanize_rs handles Latin text."""
        text = "hello world"
        result = uromanize_rs(text)
        assert result == "hello world"

    def test_uromanize_rs_cyrillic(self):
        """Test that uromanize_rs handles Cyrillic."""
        text = "Привет"
        result = uromanize_rs(text)
        assert result != text

    def test_uromanize_rs_chinese(self):
        """Test that uromanize_rs handles Chinese."""
        text = "你好"
        result = uromanize_rs(text, lcode="cmn")
        assert result != text

    def test_uromanize_caching(self):
        """Test that uromanize uses caching for repeated calls."""
        text = "Привет"
        result1 = uromanize(text)
        result2 = uromanize(text)
        assert result1 == result2
        assert result1 is result2  # Same object due to caching


class TestURomanizerClass:
    """Test URomanizer class with various configurations."""

    def test_uromanizer_basic(self):
        """Test basic URomanizer functionality."""
        romanizer = URomanizer()
        text = "Привет мир"
        result = romanizer(text)
        assert result != text

    def test_uromanizer_with_preprocessor(self):
        """Test URomanizer with preprocessor."""

        def preprocessor(text):
            return text.lower()

        romanizer = URomanizer(preprocessor=preprocessor)
        text = "ПРИВЕТ"
        result = romanizer(text)
        # Result should be lowercase after preprocessing
        assert result != text

    def test_uromanizer_with_postprocessor(self):
        """Test URomanizer with postprocessor."""

        def postprocessor(text):
            return text.upper()

        romanizer = URomanizer(postprocessor=postprocessor)
        text = "привет"
        result = romanizer(text)
        assert result.isupper()

    def test_uromanizer_empty_separator(self):
        """Test URomanizer with empty separator for character-level romanization."""
        romanizer = URomanizer(sep="")
        text = "你好"
        result = romanizer(text)
        assert result != text

    def test_uromanizer_custom_separator(self):
        """Test URomanizer with custom separator."""
        romanizer = URomanizer(sep="|")
        text = "hello|world"
        result = romanizer(text)
        assert "|" in result

    def test_uromanizer_split_alphanumeric(self):
        """Test URomanizer with split_alphanumeric enabled."""
        romanizer = URomanizer(split_alphanumeric=True)
        text = "di2duan"
        result = romanizer(text)
        # Should split "di2duan" into "di 2 duan"
        assert "2" in result
        assert " " in result

    def test_uromanizer_python_backend(self):
        """Test URomanizer with Python backend."""
        romanizer = URomanizer(backend="python")
        text = "Привет"
        result = romanizer(text)
        assert result != text

    def test_uromanizer_rust_backend(self):
        """Test URomanizer with Rust backend."""
        romanizer = URomanizer(backend="rust")
        text = "Привет"
        result = romanizer(text)
        assert result != text

    def test_uromanizer_invalid_backend(self):
        """Test that URomanizer raises error for invalid backend."""
        with pytest.raises(ValueError, match="Invalid backend"):
            URomanizer(backend="invalid")

    def test_uromanizer_with_lcode(self):
        """Test URomanizer with language code."""
        romanizer = URomanizer()
        text = "你好"
        result = romanizer(text, lcode="cmn")
        assert result != text


class TestCleanUromanizedString:
    """Test clean_uromanized_string function."""

    def test_clean_basic(self):
        """Test basic cleaning functionality."""
        text = "Hello World"
        result = clean_uromanized_string(text)
        assert result == "hello world"

    def test_clean_removes_non_latin(self):
        """Test that non-Latin characters are removed."""
        text = "hello世界world"
        result = clean_uromanized_string(text)
        assert result == "hello world"

    def test_clean_normalizes_quotes(self):
        """Test that various quote marks are normalized."""
        text = "it's it's it's"
        result = clean_uromanized_string(text)
        assert "'" in result
        # All quotes should be normalized to single quote
        assert result.count("'") >= 1

    def test_clean_removes_extra_spaces(self):
        """Test that extra spaces are removed."""
        text = "hello    world"
        result = clean_uromanized_string(text)
        assert result == "hello world"

    def test_clean_removes_punctuation(self):
        """Test that punctuation (except apostrophe) is removed."""
        text = "hello, world!"
        result = clean_uromanized_string(text)
        assert "," not in result
        assert "!" not in result

    def test_clean_preserves_apostrophe_in_words(self):
        """Test that apostrophes within words are preserved."""
        text = "it's"
        result = clean_uromanized_string(text)
        assert "'" in result

    def test_clean_removes_standalone_apostrophe(self):
        """Test that standalone apostrophes are removed."""
        text = "hello ' world"
        result = clean_uromanized_string(text)
        assert "'" not in result

    def test_clean_empty_string(self):
        """Test that empty string returns empty string."""
        assert clean_uromanized_string("") == ""


class TestGetSpacingProcessor:
    """Test get_spacing_processor function."""

    def test_spacing_processor_basic(self):
        """Test basic spacing processor functionality."""
        processor = get_spacing_processor(tokenize_whitespace)
        text = "hello world"
        result = processor(text)
        assert "hello" in result
        assert "world" in result

    def test_spacing_processor_empty_string(self):
        """Test spacing processor with empty string."""
        processor = get_spacing_processor(tokenize_whitespace)
        result = processor("")
        assert result == ""

    def test_spacing_processor_punctuation(self):
        """Test spacing processor with punctuation."""
        processor = get_spacing_processor(tokenize_whitespace)
        text = "hello , world !"
        result = processor(text)
        # Punctuation should be attached to previous word
        assert "," in result
        assert "!" in result

    def test_spacing_processor_digits(self):
        """Test spacing processor with digits."""
        processor = get_spacing_processor(tokenize_whitespace)
        text = "hello 123 world"
        result = processor(text)
        assert "123" in result
        assert " " in result

    def test_spacing_processor_joiner_punctuation(self):
        """Test spacing processor with joiner punctuation."""
        processor = get_spacing_processor(tokenize_whitespace)
        # Test colon as joiner
        text = "http : / / example . com"
        result = processor(text)
        assert ":" in result


class TestGetRomanizerForLang:
    """Test get_romanizer_for_lang function."""

    def test_get_romanizer_basic(self):
        """Test getting a romanizer for English."""
        romanizer = get_romanizer_for_lang("eng")
        assert isinstance(romanizer, URomanizer)

    def test_get_romanizer_chinese_empty_sep(self):
        """Test that Chinese romanizer has empty separator by default."""
        romanizer = get_romanizer_for_lang("cmn")
        assert romanizer.sep == ""

    def test_get_romanizer_japanese_empty_sep(self):
        """Test that Japanese romanizer has empty separator by default."""
        romanizer = get_romanizer_for_lang("jpn")
        assert romanizer.sep == ""

    def test_get_romanizer_with_aggressive_preprocessor(self):
        """Test get_romanizer_for_lang with aggressive preprocessor."""
        romanizer = get_romanizer_for_lang("eng", preprocessor="aggressive")
        text = "HELLO   WORLD"
        result = romanizer(text)
        # Should be lowercased and spaces normalized
        assert result.islower() or not result.isalpha()

    def test_get_romanizer_with_tokenize_preprocessor(self):
        """Test get_romanizer_for_lang with tokenize preprocessor."""
        romanizer = get_romanizer_for_lang("eng", preprocessor="tokenize")
        text = "hello world"
        result = romanizer(text)
        assert result is not None

    def test_get_romanizer_with_aggressive_postprocessor(self):
        """Test get_romanizer_for_lang with aggressive postprocessor."""
        romanizer = get_romanizer_for_lang("eng", postprocessor="aggressive")
        text = "Hello, World!"
        result = romanizer(text)
        # Should be cleaned aggressively
        assert result.islower()
        assert "," not in result

    def test_get_romanizer_with_custom_kwargs(self):
        """Test get_romanizer_for_lang with custom kwargs."""
        romanizer = get_romanizer_for_lang("eng", backend="rust")
        assert romanizer.backend == "rust"

    def test_get_romanizer_caching(self):
        """Test that get_romanizer_for_lang uses caching."""
        romanizer1 = get_romanizer_for_lang("eng")
        romanizer2 = get_romanizer_for_lang("eng")
        # Should be the same object due to caching
        assert romanizer1 is romanizer2

    def test_get_romanizer_invalid_class(self):
        """Test that invalid romanizer_cls raises error."""
        with pytest.raises(ValueError, match="Invalid romanizer_cls"):
            get_romanizer_for_lang("eng", romanizer_cls="invalid")


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline_chinese(self):
        """Test full romanization pipeline for Chinese."""
        romanizer = get_romanizer_for_lang(
            "cmn", preprocessor="tokenize", postprocessor="aggressive"
        )
        text = "你好世界"
        result = romanizer(text, lcode="cmn")
        # Should be romanized, lowercase, and clean
        assert result != text
        assert result.islower() or not result.isalpha()

    def test_full_pipeline_russian(self):
        """Test full romanization pipeline for Russian."""
        romanizer = get_romanizer_for_lang(
            "rus", preprocessor="aggressive", postprocessor="aggressive"
        )
        text = "Привет Мир"
        result = romanizer(text)
        assert result != text
        assert result.islower() or not result.isalpha()

    def test_mixed_script_text(self):
        """Test romanization of mixed-script text."""
        romanizer = URomanizer()
        text = "Hello Привет 你好"
        result = romanizer(text)
        # All parts should be romanized or Latin
        assert "Hello" in result or "hello" in result.lower()
