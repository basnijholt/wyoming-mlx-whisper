"""Tests for the const module."""

from wyoming_mlx_whisper.const import WHISPER_LANGUAGES


class TestWhisperLanguages:
    """Tests for WHISPER_LANGUAGES constant."""

    def test_is_list(self) -> None:
        """Test that WHISPER_LANGUAGES is a list."""
        assert isinstance(WHISPER_LANGUAGES, list)

    def test_contains_common_languages(self) -> None:
        """Test that common languages are included."""
        common = ["en", "es", "fr", "de", "ja", "zh", "ko", "pt", "it", "ru"]
        for lang in common:
            assert lang in WHISPER_LANGUAGES, f"Missing language: {lang}"

    def test_all_lowercase(self) -> None:
        """Test that all language codes are lowercase."""
        for lang in WHISPER_LANGUAGES:
            assert lang == lang.lower(), f"Language code not lowercase: {lang}"

    def test_no_duplicates(self) -> None:
        """Test that there are no duplicate language codes."""
        assert len(WHISPER_LANGUAGES) == len(set(WHISPER_LANGUAGES))

    def test_reasonable_count(self) -> None:
        """Test that there's a reasonable number of languages."""
        # Whisper supports ~99 languages
        assert len(WHISPER_LANGUAGES) >= 90
        assert len(WHISPER_LANGUAGES) <= 120
