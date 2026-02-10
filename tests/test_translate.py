"""
Unit tests for the unified translate() function.
Tests all 3 supported models and error handling.
"""

import pytest

from muse.translation.translate import translate


class TestTranslateErrorHandling:
    """Test error handling for the translate() function."""

    def test_unsupported_model_raises_error(self):
        """Test that unsupported model raises ValueError with helpful message."""
        with pytest.raises(ValueError, match="Unsupported model"):
            translate("unsupported/model", "en", "es", "hello")

    def test_invalid_source_language_raises_error(self):
        """Test that invalid source language raises ValueError."""
        with pytest.raises(ValueError, match=r"Source language .* is not supported"):
            translate("tencent/HY-MT1.5-7B", "invalid_lang", "en", "test")

    def test_invalid_target_language_raises_error(self):
        """Test that invalid target language raises ValueError."""
        with pytest.raises(ValueError, match=r"Target language .* is not supported"):
            translate("tencent/HY-MT1.5-7B", "en", "invalid_lang", "test")


class TestTranslateHYMT:
    """Test translations using the HY-MT model."""

    def test_chinese_to_english(self):
        """Test Chinese to English translation with HY-MT."""
        result = translate(
            model="tencent/HY-MT1.5-7B",
            src_lang="zh",
            tgt_lang="en",
            text="音乐理论",
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_english_to_chinese(self):
        """Test English to Chinese translation with HY-MT."""
        result = translate(
            model="tencent/HY-MT1.5-7B",
            src_lang="en",
            tgt_lang="zh",
            text="Hello, how are you?",
        )
        assert isinstance(result, str)
        assert len(result) > 0


class TestTranslateNLLB:
    """Test translations using the NLLB model."""

    def test_japanese_to_english(self):
        """Test Japanese to English translation with NLLB."""
        result = translate(
            model="facebook/nllb-200-3.3B",
            src_lang="ja",
            tgt_lang="en",
            text="音楽理論",
        )
        assert isinstance(result, str)
        assert len(result) > 0


class TestTranslateMADLAD:
    """Test translations using the MADLAD model."""

    def test_english_to_spanish(self):
        """Test English to Spanish translation with MADLAD."""
        result = translate(
            model="google/madlad400-7b-mt",
            src_lang="en",  # Accepted but not used by MADLAD
            tgt_lang="es",
            text="music theory",
        )
        assert isinstance(result, str)
        assert len(result) > 0
