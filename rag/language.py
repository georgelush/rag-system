"""
rag/language.py — Language detection and ISO 639-1 utilities.

Provides:
  - detect_language(text) -> str   — returns ISO 639-1 code, e.g. "en", "ro", "de"
  - language_name(code)   -> str   — returns English name for system prompts, e.g. "German"

Detection is done locally using lingua-language-detector (no external API call).
Falls back to "en" when detection confidence is too low or text is too short.
"""

from lingua import Language, LanguageDetectorBuilder

# Build detector once at module load — expensive operation (~200ms), singleton pattern.
# Covers the 75 most common languages with high accuracy on short texts (≥ 5 words).
_detector = (
    LanguageDetectorBuilder
    .from_all_languages()
    .with_minimum_relative_distance(0.1)  # confidence threshold — below this → fallback
    .build()
)

# ISO 639-1 code → English language name (used in system prompts)
_CODE_TO_NAME: dict[str, str] = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "az": "Azerbaijani",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bs": "Bosnian",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "eo": "Esperanto",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "ga": "Irish",
    "gl": "Galician",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "ko": "Korean",
    "la": "Latin",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "nb": "Norwegian Bokmål",
    "nl": "Dutch",
    "nn": "Norwegian Nynorsk",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sq": "Albanian",
    "sr": "Serbian",
    "sv": "Swedish",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Filipino",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "zh": "Chinese",
}

_FALLBACK = "en"


def detect_language(text: str) -> str:
    """Detect the language of text and return an ISO 639-1 code.

    Args:
        text: Plain text to analyse. Works best with ≥ 10 words.

    Returns:
        ISO 639-1 code (e.g. "en", "ro", "de").
        Falls back to "en" when text is too short or confidence is too low.
    """
    if not text or len(text.split()) < 3:
        return _FALLBACK

    detected: Language | None = _detector.detect_language_of(text[:2000])

    if detected is None:
        return _FALLBACK

    # lingua uses its own Language enum — extract ISO code via .iso_code_639_1
    try:
        code = detected.iso_code_639_1.name.lower()
    except AttributeError:
        return _FALLBACK

    return code if code in _CODE_TO_NAME else _FALLBACK


def language_name(code: str) -> str:
    """Return the English name of a language given its ISO 639-1 code.

    Used to build system prompts like "Respond in German."

    Args:
        code: ISO 639-1 code, e.g. "de".

    Returns:
        English name, e.g. "German". Falls back to "English" for unknown codes.
    """
    return _CODE_TO_NAME.get(code.lower(), "English")
