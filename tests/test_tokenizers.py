"""Tests for tokenizers module."""

import pytest
from trom.tokenizers import (
    get_moses_tokenizer,
    get_spacy_tokenizer,
    get_tokenizer_for_lang,
    remove_whitespaces,
    tokenize_cantonese,
    tokenize_chinese,
    tokenize_thai,
    tokenize_whitespace,
    tokenizer_khmer,
    word_tokenize,
)


class TestBasicTokenizers:
    """Test basic tokenizer functions."""

    def test_word_tokenize_basic(self):
        """Test basic word tokenization with regex."""
        text = "hello world"
        result = word_tokenize(text)
        assert result == ["hello", "world"]

    def test_word_tokenize_with_punctuation(self):
        """Test word tokenization ignores punctuation."""
        text = "hello, world!"
        result = word_tokenize(text)
        assert result == ["hello", "world"]

    def test_word_tokenize_with_numbers(self):
        """Test word tokenization with numbers."""
        text = "hello 123 world"
        result = word_tokenize(text)
        assert result == ["hello", "123", "world"]

    def test_word_tokenize_empty(self):
        """Test word tokenization with empty string."""
        result = word_tokenize("")
        assert result == []

    def test_remove_whitespaces(self):
        """Test removing all whitespaces."""
        text = "hello world"
        result = remove_whitespaces(text)
        assert result == "helloworld"

    def test_remove_whitespaces_multiple(self):
        """Test removing multiple types of whitespaces."""
        text = "hello  \t\n world"
        result = remove_whitespaces(text)
        assert result == "helloworld"

    def test_remove_whitespaces_empty(self):
        """Test removing whitespaces from empty string."""
        result = remove_whitespaces("")
        assert result == ""


class TestWhitespaceTokenizer:
    """Test whitespace tokenizer."""

    def test_tokenize_whitespace_basic(self):
        """Test basic whitespace tokenization."""
        text = "hello world"
        result = tokenize_whitespace(text)
        assert result == ["hello", "world"]

    def test_tokenize_whitespace_multiple_spaces(self):
        """Test whitespace tokenization with multiple spaces."""
        text = "hello   world"
        result = tokenize_whitespace(text)
        assert result == ["hello", "world"]

    def test_tokenize_whitespace_tabs_newlines(self):
        """Test whitespace tokenization with tabs and newlines."""
        text = "hello\t\nworld"
        result = tokenize_whitespace(text)
        assert result == ["hello", "world"]

    def test_tokenize_whitespace_leading_trailing(self):
        """Test whitespace tokenization removes leading/trailing spaces."""
        text = "  hello world  "
        result = tokenize_whitespace(text)
        assert result == ["hello", "world"]

    def test_tokenize_whitespace_empty(self):
        """Test whitespace tokenization with empty string."""
        result = tokenize_whitespace("")
        assert result == []

    def test_tokenize_whitespace_preserves_punctuation(self):
        """Test that whitespace tokenizer preserves punctuation."""
        text = "hello, world!"
        result = tokenize_whitespace(text)
        assert result == ["hello,", "world!"]


class TestLanguageSpecificTokenizers:
    """Test language-specific tokenizers."""

    def test_tokenize_chinese_basic(self):
        """Test Chinese tokenization."""
        text = "你好世界"
        result = tokenize_chinese(text)
        assert isinstance(result, list)
        assert len(result) > 0
        # Jieba should split this into meaningful units
        assert all(isinstance(token, str) for token in result)

    def test_tokenize_chinese_mixed(self):
        """Test Chinese tokenization with mixed content."""
        text = "hello你好"
        result = tokenize_chinese(text)
        assert isinstance(result, list)
        assert len(result) >= 2

    def test_tokenize_cantonese_basic(self):
        """Test Cantonese tokenization."""
        text = "你好"
        result = tokenize_cantonese(text)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_tokenize_thai_basic(self):
        """Test Thai tokenization."""
        text = "สวัสดี"
        result = tokenize_thai(text)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_tokenizer_khmer_basic(self):
        """Test Khmer tokenization."""
        text = "សួស្តី"
        result = tokenizer_khmer(text)
        assert isinstance(result, list)
        assert len(result) > 0


class TestMosesTokenizer:
    """Test Moses tokenizer."""

    def test_get_moses_tokenizer_english(self):
        """Test getting Moses tokenizer for English."""
        tokenizer = get_moses_tokenizer("en")
        text = "Hello, world!"
        result = tokenizer(text)
        assert isinstance(result, list)
        assert "Hello" in result or "hello" in result
        # Moses should separate punctuation
        assert "," in result or ",".join(result) != text

    def test_get_moses_tokenizer_no_lang(self):
        """Test getting Moses tokenizer without language specification."""
        tokenizer = get_moses_tokenizer(None)
        text = "Hello world"
        result = tokenizer(text)
        assert isinstance(result, list)

    def test_moses_tokenizer_punctuation(self):
        """Test that Moses tokenizer separates punctuation."""
        tokenizer = get_moses_tokenizer("en")
        text = "Hello, world!"
        result = tokenizer(text)
        # Punctuation should be separate tokens
        assert len(result) >= 3

    def test_moses_tokenizer_contractions(self):
        """Test that Moses tokenizer handles contractions."""
        tokenizer = get_moses_tokenizer("en")
        text = "it's"
        result = tokenizer(text)
        assert isinstance(result, list)
        # Moses may split contractions
        assert len(result) >= 1


class TestSpacyTokenizer:
    """Test spaCy tokenizer."""

    def test_get_spacy_tokenizer_japanese(self):
        """Test getting spaCy tokenizer for Japanese."""
        tokenizer = get_spacy_tokenizer("jpn")
        text = "こんにちは世界"
        result = tokenizer(text)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_get_spacy_tokenizer_korean(self):
        """Test getting spaCy tokenizer for Korean."""
        tokenizer = get_spacy_tokenizer("kor")
        text = "안녕하세요"
        result = tokenizer(text)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_get_spacy_tokenizer_chinese(self):
        """Test getting spaCy tokenizer for Chinese."""
        tokenizer = get_spacy_tokenizer("cmn")
        text = "你好世界"
        result = tokenizer(text)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_get_spacy_tokenizer_unsupported_lang(self):
        """Test that unsupported language raises error."""
        with pytest.raises(ValueError, match="not found in ISO 639"):
            get_spacy_tokenizer("xyz")

    def test_spacy_tokenizer_filters_spaces(self):
        """Test that spaCy tokenizer filters out space tokens."""
        tokenizer = get_spacy_tokenizer("jpn")
        text = "こんにちは 世界"
        result = tokenizer(text)
        # Should not include space-only tokens
        assert all(token.strip() for token in result)


class TestGetTokenizerForLang:
    """Test get_tokenizer_for_lang function."""

    def test_get_tokenizer_khmer(self):
        """Test getting tokenizer for Khmer."""
        tokenizer = get_tokenizer_for_lang("khm")
        text = "សួស្តី"
        result = tokenizer(text)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_get_tokenizer_chinese(self):
        """Test getting tokenizer for Chinese."""
        tokenizer = get_tokenizer_for_lang("cmn")
        text = "你好"
        result = tokenizer(text)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_get_tokenizer_thai(self):
        """Test getting tokenizer for Thai."""
        tokenizer = get_tokenizer_for_lang("tha")
        text = "สวัสดี"
        result = tokenizer(text)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_get_tokenizer_french(self):
        """Test getting tokenizer for French."""
        tokenizer = get_tokenizer_for_lang("fra")
        text = "Bonjour le monde"
        result = tokenizer(text)
        assert isinstance(result, list)
        # French uses whitespace tokenizer
        assert "Bonjour" in result

    def test_get_tokenizer_english(self):
        """Test getting tokenizer for English."""
        tokenizer = get_tokenizer_for_lang("eng")
        text = "Hello world"
        result = tokenizer(text)
        assert isinstance(result, list)
        # English uses Moses tokenizer
        assert len(result) >= 2

    def test_get_tokenizer_japanese(self):
        """Test getting tokenizer for Japanese."""
        tokenizer = get_tokenizer_for_lang("jpn")
        text = "こんにちは"
        result = tokenizer(text)
        assert isinstance(result, list)

    def test_get_tokenizer_korean(self):
        """Test getting tokenizer for Korean."""
        tokenizer = get_tokenizer_for_lang("kor")
        text = "안녕하세요"
        result = tokenizer(text)
        assert isinstance(result, list)

    def test_get_tokenizer_unsupported_raises_error(self):
        """Test that unsupported language raises ValueError."""
        with pytest.raises(ValueError, match="not found in ISO 639"):
            get_tokenizer_for_lang("xyz")

    def test_get_tokenizer_unsupported_raises_error_with_moses(self):
        """Test that unsupported language raises ValueError even with default_moses=True."""
        with pytest.raises(ValueError, match="not found in ISO 639"):
            get_tokenizer_for_lang("xyz", default_moses=True)

    def test_get_tokenizer_caching(self):
        """Test that get_tokenizer_for_lang uses caching."""
        tokenizer1 = get_tokenizer_for_lang("eng")
        tokenizer2 = get_tokenizer_for_lang("eng")
        # Should return same function object due to caching
        assert tokenizer1 is tokenizer2

    def test_get_tokenizer_cantonese(self):
        """Test getting tokenizer for Cantonese."""
        tokenizer = get_tokenizer_for_lang("yue")
        text = "你好"
        result = tokenizer(text)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_get_tokenizer_lao(self):
        """Test getting tokenizer for Lao."""
        tokenizer = get_tokenizer_for_lang("lao")
        text = "ສະບາຍດີ"
        result = tokenizer(text)
        assert isinstance(result, list)
        assert len(result) > 0


class TestIntegration:
    """Integration tests for tokenizers."""

    def test_tokenizer_with_empty_input(self):
        """Test that tokenizers handle empty input gracefully."""
        for lang in ["eng", "cmn", "jpn", "fra"]:
            tokenizer = get_tokenizer_for_lang(lang)
            result = tokenizer("")
            assert isinstance(result, list)

    def test_tokenizer_with_whitespace_only(self):
        """Test that tokenizers handle whitespace-only input."""
        tokenizer = get_tokenizer_for_lang("eng")
        result = tokenizer("   \t\n   ")
        # Should return empty list or list with empty strings
        assert isinstance(result, list)

    def test_tokenizer_consistency(self):
        """Test that tokenizers return consistent results."""
        tokenizer = get_tokenizer_for_lang("eng")
        text = "Hello world"
        result1 = tokenizer(text)
        result2 = tokenizer(text)
        assert result1 == result2

    def test_tokenizer_with_numbers_and_symbols(self):
        """Test tokenizers with mixed content."""
        tokenizer = get_tokenizer_for_lang("eng")
        text = "Hello 123 @ world!"
        result = tokenizer(text)
        assert isinstance(result, list)
        assert len(result) > 0
