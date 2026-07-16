"""Context extractionで使うlanguage tagの正規化。"""

_LANGUAGE_ALIASES = {"jpn": "ja", "eng": "en"}


def normalize_context_language(language: str) -> str:
    """BCP 47相当tagをbackend向けprimary language codeへ正規化する。"""
    primary = language.strip().lower().replace("_", "-").split("-", maxsplit=1)[0]
    return _LANGUAGE_ALIASES.get(primary, primary)


def normalize_optional_stream_language(language: str | None) -> str | None:
    """任意stream metadataを比較可能なprimary languageへ正規化する。"""
    if language is None:
        return None
    normalized = normalize_context_language(language)
    return None if not normalized or normalized == "und" else normalized
