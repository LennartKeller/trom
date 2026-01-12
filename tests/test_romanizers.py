"""Tests for romanizers module."""

import pytest

from trom.romanizers import (
    UnifiedRomanizer,
    URomanizer,
    clean_uromanized_string,
    get_romanizer_for_lang,
    get_spacing_processor,
    uromanize,
    uromanize_rs,
)
from trom.tokenizers import tokenize_whitespace
from trom.unicode import detect_script_spans, get_script, is_latin_only


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


class TestScriptDetection:
    """Test script detection utilities."""

    def test_get_script_latin(self):
        """Test script detection for Latin characters."""
        assert get_script("a") == "Latin"
        assert get_script("Z") == "Latin"
        assert get_script("é") == "Latin"

    def test_get_script_cyrillic(self):
        """Test script detection for Cyrillic characters."""
        assert get_script("П") == "Cyrillic"
        assert get_script("ж") == "Cyrillic"

    def test_get_script_greek(self):
        """Test script detection for Greek characters."""
        assert get_script("Γ") == "Greek"
        assert get_script("α") == "Greek"

    def test_get_script_han(self):
        """Test script detection for Han (Chinese) characters."""
        assert get_script("你") == "Han"
        assert get_script("好") == "Han"

    def test_get_script_hiragana(self):
        """Test script detection for Hiragana."""
        assert get_script("あ") == "Hiragana"
        assert get_script("の") == "Hiragana"

    def test_get_script_katakana(self):
        """Test script detection for Katakana."""
        assert get_script("ア") == "Katakana"
        assert get_script("カ") == "Katakana"

    def test_get_script_hangul(self):
        """Test script detection for Hangul (Korean)."""
        assert get_script("안") == "Hangul"
        assert get_script("녕") == "Hangul"

    def test_get_script_arabic(self):
        """Test script detection for Arabic."""
        assert get_script("م") == "Arabic"
        assert get_script("ر") == "Arabic"

    def test_get_script_cherokee(self):
        """Test script detection for Cherokee."""
        assert get_script("Ꭰ") == "Cherokee"
        assert get_script("Ꮧ") == "Cherokee"

    def test_get_script_common(self):
        """Test script detection for common characters."""
        assert get_script(" ") == "Common"
        assert get_script("1") == "Common"
        assert get_script(".") == "Common"
        assert get_script(",") == "Common"

    def test_detect_script_spans_single_script(self):
        """Test script span detection for single-script text."""
        spans = detect_script_spans("Hello")
        assert len(spans) == 1
        assert spans[0][0] == "Latin"
        assert spans[0][1] == 0
        assert spans[0][2] == 5

    def test_detect_script_spans_mixed(self):
        """Test script span detection for mixed-script text."""
        spans = detect_script_spans("Hello你好")
        assert len(spans) == 2
        assert spans[0][0] == "Latin"
        assert spans[1][0] == "Han"

    def test_detect_script_spans_with_punctuation(self):
        """Test script span detection with punctuation."""
        spans = detect_script_spans("Hello, 你好!")
        # Common (punctuation) should be merged with adjacent scripts
        assert len(spans) >= 2

    def test_detect_script_spans_empty(self):
        """Test script span detection for empty string."""
        spans = detect_script_spans("")
        assert len(spans) == 0


class TestUnifiedRomanizer:
    """Test UnifiedRomanizer class."""

    def test_unified_romanizer_basic(self):
        """Test basic UnifiedRomanizer initialization."""
        romanizer = UnifiedRomanizer()
        assert romanizer is not None
        assert len(romanizer.backends) > 0

    def test_unified_romanizer_latin_passthrough(self):
        """Test that Latin text passes through unchanged."""
        romanizer = UnifiedRomanizer()
        text = "Hello World"
        result = romanizer(text)
        assert result == "Hello World"

    def test_unified_romanizer_cyrillic(self):
        """Test Cyrillic romanization."""
        romanizer = UnifiedRomanizer()
        text = "Привет"
        result = romanizer(text)
        assert result != text
        assert result.lower().startswith("p")  # Should start with 'p' for Privet

    def test_unified_romanizer_greek(self):
        """Test Greek romanization."""
        romanizer = UnifiedRomanizer()
        text = "Γεια"
        result = romanizer(text)
        assert result != text
        # Greek Gamma can be romanized as G or Y
        assert result[0].lower() in ("g", "y")

    def test_unified_romanizer_chinese(self):
        """Test Chinese romanization."""
        romanizer = UnifiedRomanizer()
        text = "你好"
        result = romanizer(text, lcode="cmn")
        assert result != text
        assert len(result) > 0

    def test_unified_romanizer_arabic(self):
        """Test Arabic romanization."""
        romanizer = UnifiedRomanizer()
        text = "مرحبا"
        result = romanizer(text)
        assert result != text
        assert len(result) > 0

    def test_unified_romanizer_mixed_latin_cyrillic(self):
        """Test mixed Latin and Cyrillic text."""
        romanizer = UnifiedRomanizer()
        text = "Hello Привет"
        result = romanizer(text)
        assert "Hello" in result or "hello" in result.lower()
        # Cyrillic part should be romanized
        assert result != text

    def test_unified_romanizer_mixed_latin_chinese(self):
        """Test mixed Latin and Chinese text."""
        romanizer = UnifiedRomanizer()
        text = "Hello 你好"
        result = romanizer(text, lcode="cmn")
        assert "Hello" in result or "hello" in result.lower()
        # Chinese part should be romanized
        assert result != text

    def test_unified_romanizer_mixed_three_scripts(self):
        """Test text with three different scripts."""
        romanizer = UnifiedRomanizer()
        text = "Hello Привет 你好"
        result = romanizer(text, lcode="cmn")
        assert "Hello" in result or "hello" in result.lower()
        # Both non-Latin parts should be romanized
        assert result != text
        assert len(result) > len("Hello")

    def test_unified_romanizer_empty_string(self):
        """Test empty string handling."""
        romanizer = UnifiedRomanizer()
        result = romanizer("")
        assert result == ""

    def test_unified_romanizer_whitespace_only(self):
        """Test whitespace-only string."""
        romanizer = UnifiedRomanizer()
        result = romanizer("   ")
        assert result == "   "

    def test_unified_romanizer_with_preprocessor(self):
        """Test UnifiedRomanizer with preprocessor."""

        def preprocessor(text):
            return text.lower()

        romanizer = UnifiedRomanizer(preprocessor=preprocessor)
        text = "HELLO WORLD"
        result = romanizer(text)
        assert result == "hello world"

    def test_unified_romanizer_with_postprocessor(self):
        """Test UnifiedRomanizer with postprocessor."""

        def postprocessor(text):
            return text.upper()

        romanizer = UnifiedRomanizer(postprocessor=postprocessor)
        text = "hello"
        result = romanizer(text)
        assert result == "HELLO"

    def test_unified_romanizer_empty_separator(self):
        """Test UnifiedRomanizer with empty separator (character-level)."""
        romanizer = UnifiedRomanizer(sep="")
        text = "你好"
        result = romanizer(text, lcode="cmn")
        assert result != text
        assert len(result) > 0

    def test_unified_romanizer_custom_separator(self):
        """Test UnifiedRomanizer with custom separator."""
        romanizer = UnifiedRomanizer(sep="|")
        text = "Hello|Привет"
        result = romanizer(text)
        assert "|" in result
        assert "Hello" in result

    def test_unified_romanizer_backend_preference_auto(self):
        """Test auto backend preference."""
        romanizer = UnifiedRomanizer(backend_preference="auto")
        assert len(romanizer.backend_preference) > 0

    def test_unified_romanizer_backend_preference_uroman(self):
        """Test uroman backend preference."""
        romanizer = UnifiedRomanizer(backend_preference="uroman")
        assert "uroman" in romanizer.backend_preference

    def test_unified_romanizer_uroman_rust_backend(self):
        """Test UnifiedRomanizer with Rust uroman backend."""
        romanizer = UnifiedRomanizer(uroman_backend="rust")
        assert "uroman" in romanizer.backends
        text = "Привет"
        result = romanizer(text)
        assert result != text

    def test_unified_romanizer_japanese_mixed(self):
        """Test Japanese text with mixed scripts (Hiragana, Katakana, Kanji)."""
        romanizer = UnifiedRomanizer(sep="")
        text = "こんにちは"  # Hiragana
        result = romanizer(text, lcode="jpn")
        assert result != text
        assert len(result) > 0

    def test_unified_romanizer_korean(self):
        """Test Korean romanization."""
        romanizer = UnifiedRomanizer()
        text = "안녕하세요"
        result = romanizer(text, lcode="kor")
        assert result != text
        assert len(result) > 0

    def test_unified_romanizer_cherokee(self):
        """Test Cherokee romanization (ICU doesn't support, should fallback to uroman)."""
        romanizer = UnifiedRomanizer()
        text = "ᎠᎢᎣᎤᎥ ᎦᏬᏂᎯᏍᏗ"
        result = romanizer(text)
        assert result != text
        # Cherokee should be romanized by uroman since ICU doesn't support it
        assert "Ꭰ" not in result
        assert len(result) > 0

    def test_unified_romanizer_punctuation_preserved(self):
        """Test that punctuation is preserved."""
        romanizer = UnifiedRomanizer()
        text = "Hello, world!"
        result = romanizer(text)
        assert "," in result
        assert "!" in result

    def test_unified_romanizer_numbers_preserved(self):
        """Test that numbers are preserved."""
        romanizer = UnifiedRomanizer()
        text = "Hello 123 world"
        result = romanizer(text)
        assert "123" in result

    def test_unified_romanizer_verbose_mode(self):
        """Test UnifiedRomanizer with verbose mode."""
        romanizer = UnifiedRomanizer(verbose=True)
        assert romanizer.verbose is True
        # Just ensure it doesn't crash
        text = "Hello"
        result = romanizer(text)
        assert result == "Hello"
    
    def test_unified_romanizer_interleaved_scripts(self):
        """Test UnifiedRomanizer with interleaved scripts."""
        romanizer = UnifiedRomanizer()
        text = """
ᎠᎢᎣᎤᎥ 今天天气很好 một ngày tồi tệ ᎦᏬᏂᎯᏍᏗ. だけど、世界和平 những ngôi sao xa xôi ᎡᎶᎯ ᎢᎦ. 繁荣昌盛 sự phát triển bền vững ᎣᏍᏓ ᏑᎾᎴᎢ 图书馆. ᏫᏥᎦᎵᎡᎵᎦ cái cây xanh 电脑 互相理解.

ᎾᎿᎢ ᎢᏳᏍᏗ 我们去哪儿? Con đường dài ᎡᎵᏍᏗ ᎪᎯ ᎢᎦ. 苹果和橘子 hạnh phúc là gì ᏍᎩ ᏗᏕᎶᏆᏍᏗ. 什么是爱 không biết nữa ᏣᎳᎩ. 天地玄黄 những giấc mơ kỳ lạ ᎾᏍᎩ ᎢᏳᏍᏗ.

ᎣᏍᏓ ᎢᎦ 早上好 tạm biệt nhé ᎦᎵᎡᎵᎦ. 语言障碍 khó khăn chồng chất ᎡᎶᎯ. 飞机场 ở đâu đó ᏑᎾᎴᎢ. 很高兴认识你 rất vui được gặp bạn ᏫᏥᎦᎵᎡᎵᎦ.
"""
        result = romanizer(text)
        assert is_latin_only(result)


class TestGetRomanizerForLangUnified:
    """Test get_romanizer_for_lang with unified romanizer."""

    def test_get_unified_romanizer(self):
        """Test getting a unified romanizer."""
        romanizer = get_romanizer_for_lang("eng", romanizer_cls="unified")
        assert isinstance(romanizer, UnifiedRomanizer)

    def test_unified_romanizer_with_aggressive_preprocessor(self):
        """Test unified romanizer with aggressive preprocessor."""
        romanizer = get_romanizer_for_lang(
            "eng", romanizer_cls="unified", preprocessor="aggressive"
        )
        text = "HELLO   WORLD"
        result = romanizer(text)
        assert result.islower() or not result.isalpha()

    def test_unified_romanizer_with_tokenize_preprocessor(self):
        """Test unified romanizer with tokenize preprocessor."""
        romanizer = get_romanizer_for_lang(
            "eng", romanizer_cls="unified", preprocessor="tokenize"
        )
        text = "hello world"
        result = romanizer(text)
        assert result is not None

    def test_unified_romanizer_with_aggressive_postprocessor(self):
        """Test unified romanizer with aggressive postprocessor."""
        romanizer = get_romanizer_for_lang(
            "eng", romanizer_cls="unified", postprocessor="aggressive"
        )
        text = "Hello, World!"
        result = romanizer(text)
        assert result.islower()
        assert "," not in result

    def test_unified_romanizer_chinese_config(self):
        """Test unified romanizer with Chinese language config."""
        romanizer = get_romanizer_for_lang("cmn", romanizer_cls="unified")
        # Should have empty separator for Chinese
        assert romanizer.sep == ""

    def test_unified_romanizer_japanese_config(self):
        """Test unified romanizer with Japanese language config."""
        romanizer = get_romanizer_for_lang("jpn", romanizer_cls="unified")
        # Should have empty separator for Japanese
        assert romanizer.sep == ""

    def test_unified_romanizer_caching(self):
        """Test that get_romanizer_for_lang caches unified romanizers."""
        romanizer1 = get_romanizer_for_lang("eng", romanizer_cls="unified")
        romanizer2 = get_romanizer_for_lang("eng", romanizer_cls="unified")
        # Should be the same object due to caching
        assert romanizer1 is romanizer2


class TestUnifiedRomanizerEdgeCases:
    """Test edge cases and error handling for UnifiedRomanizer."""

    def test_unified_romanizer_only_punctuation(self):
        """Test text with only punctuation."""
        romanizer = UnifiedRomanizer()
        text = "... !!! ???"
        result = romanizer(text)
        # Should pass through unchanged
        assert result == text

    def test_unified_romanizer_mixed_with_numbers(self):
        """Test mixed script text with numbers."""
        romanizer = UnifiedRomanizer()
        text = "abc123你好456xyz"
        result = romanizer(text, lcode="cmn")
        # Latin and numbers should be preserved
        assert "abc" in result or "ABC" in result.lower()
        assert "123" in result
        assert "456" in result
        assert "xyz" in result or "XYZ" in result.lower()

    def test_unified_romanizer_consecutive_spaces(self):
        """Test that consecutive spaces are handled properly."""
        romanizer = UnifiedRomanizer()
        text = "Hello    World"
        result = romanizer(text)
        # Spaces should be normalized
        assert "Hello" in result
        assert "World" in result

    def test_unified_romanizer_newlines(self):
        """Test text with newlines."""
        romanizer = UnifiedRomanizer()
        text = "Hello\nWorld"
        result = romanizer(text)
        # Should handle newlines gracefully
        assert "Hello" in result or "hello" in result.lower()
        assert "World" in result or "world" in result.lower()

    def test_unified_romanizer_tabs(self):
        """Test text with tabs."""
        romanizer = UnifiedRomanizer()
        text = "Hello\tWorld"
        result = romanizer(text)
        assert "Hello" in result or "hello" in result.lower()
        assert "World" in result or "world" in result.lower()
