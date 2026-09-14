"""Language definitions for the gTTS-voices benchmark.

The "models" are the 69 gTTS voices (IETF tags).  The languages they are
asked to read are the 12 Ghanaian languages, whose graphemes are first
universalised with africa-g2p.
"""

# ── The 69 gTTS voices (exactly what the installed gTTS supports) ──────────
def _gtts_langs():
    try:
        from gtts.lang import tts_langs

        return dict(tts_langs())
    except Exception:
        pass
    # Fallback (never reached in normal use).
    return {
        "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "bg": "Bulgarian",
        "bn": "Bengali", "bs": "Bosnian", "ca": "Catalan", "cs": "Czech",
        "cy": "Welsh", "da": "Danish", "de": "German", "el": "Greek",
        "en": "English", "eo": "Esperanto", "es": "Spanish", "et": "Estonian",
        "eu": "Basque", "fi": "Finnish", "fr": "French", "fr-CA": "French (Canada)",
        "gl": "Galician", "gu": "Gujarati", "hi": "Hindi", "hr": "Croatian",
        "hu": "Hungarian", "hy": "Armenian", "id": "Indonesian",
        "is": "Icelandic", "it": "Italian", "ja": "Japanese", "jw": "Javanese",
        "km": "Khmer", "kn": "Kannada", "ko": "Korean", "la": "Latin",
        "lt": "Lithuanian", "lv": "Latvian", "mk": "Macedonian",
        "ml": "Malayalam", "mr": "Marathi", "ms": "Malay", "mt": "Maltese",
        "my": "Myanmar", "ne": "Nepali", "nl": "Dutch", "no": "Norwegian",
        "pl": "Polish", "pt": "Portuguese", "pt-PT": "Portuguese (Portugal)",
        "ro": "Romanian", "ru": "Russian", "si": "Sinhala", "sk": "Slovak",
        "sq": "Albanian", "sr": "Serbian", "su": "Sundanese", "sv": "Swedish",
        "sw": "Swahili", "ta": "Tamil", "te": "Telugu", "th": "Thai",
        "tl": "Filipino", "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu",
        "vi": "Vietnamese", "zh-CN": "Chinese (Simplified)",
        "zh-TW": "Chinese (Traditional)", "zh": "Chinese (Mandarin)",
        "zu": "Zulu", "ha": "Hausa", "pa": "Punjabi", "yue": "Cantonese",
        "iw": "Hebrew",
    }


GTTS_LANGS = _gtts_langs()

# gTTS voices that differ from their base code (variants on the board).
VOICE_VARIANTS = {"fr-CA", "pt-PT", "zh-CN", "zh-TW", "zh"}


def voices():
    return sorted(GTTS_LANGS)


def voice_name(ietf):
    return GTTS_LANGS.get(ietf, ietf)