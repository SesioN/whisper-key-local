from typing import Optional

AUTO_LANGUAGE = "auto"

WHISPER_LANGUAGE_NAMES = {
    "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "as": "Assamese", "az": "Azerbaijani", "ba": "Bashkir",
    "be": "Belarusian", "bg": "Bulgarian", "bn": "Bengali", "bo": "Tibetan", "br": "Breton", "bs": "Bosnian",
    "ca": "Catalan", "cs": "Czech", "cy": "Welsh", "da": "Danish", "de": "German", "el": "Greek", "en": "English",
    "es": "Spanish", "et": "Estonian", "eu": "Basque", "fa": "Persian", "fi": "Finnish", "fo": "Faroese",
    "fr": "French", "gl": "Galician", "gu": "Gujarati", "ha": "Hausa", "haw": "Hawaiian", "he": "Hebrew",
    "hi": "Hindi", "hr": "Croatian", "ht": "Haitian Creole", "hu": "Hungarian", "hy": "Armenian",
    "id": "Indonesian", "is": "Icelandic", "it": "Italian", "ja": "Japanese", "jw": "Javanese", "ka": "Georgian",
    "kk": "Kazakh", "km": "Khmer", "kn": "Kannada", "ko": "Korean", "la": "Latin", "lb": "Luxembourgish",
    "ln": "Lingala", "lo": "Lao", "lt": "Lithuanian", "lv": "Latvian", "mg": "Malagasy", "mi": "Maori",
    "mk": "Macedonian", "ml": "Malayalam", "mn": "Mongolian", "mr": "Marathi", "ms": "Malay", "mt": "Maltese",
    "my": "Myanmar", "ne": "Nepali", "nl": "Dutch", "nn": "Nynorsk", "no": "Norwegian", "oc": "Occitan",
    "pa": "Punjabi", "pl": "Polish", "ps": "Pashto", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian",
    "sa": "Sanskrit", "sd": "Sindhi", "si": "Sinhala", "sk": "Slovak", "sl": "Slovenian", "sn": "Shona",
    "so": "Somali", "sq": "Albanian", "sr": "Serbian", "su": "Sundanese", "sv": "Swedish", "sw": "Swahili",
    "ta": "Tamil", "te": "Telugu", "tg": "Tajik", "th": "Thai", "tk": "Turkmen", "tl": "Tagalog (Filipino)", "tr": "Turkish",
    "tt": "Tatar", "uk": "Ukrainian", "ur": "Urdu", "uz": "Uzbek", "vi": "Vietnamese", "yi": "Yiddish",
    "yo": "Yoruba", "zh": "Chinese", "yue": "Cantonese",
}

QWEN_LANGUAGE_NAMES = {
    "ar": "Arabic", "yue": "Cantonese", "zh": "Chinese", "cs": "Czech", "da": "Danish", "nl": "Dutch",
    "en": "English", "fil": "Filipino", "tl": "Filipino", "fi": "Finnish", "fr": "French", "de": "German",
    "el": "Greek", "hi": "Hindi", "hu": "Hungarian", "id": "Indonesian", "it": "Italian", "ja": "Japanese",
    "ko": "Korean", "mk": "Macedonian", "ms": "Malay", "fa": "Persian", "pl": "Polish", "pt": "Portuguese",
    "ro": "Romanian", "ru": "Russian", "es": "Spanish", "sv": "Swedish", "th": "Thai", "tr": "Turkish",
    "vi": "Vietnamese",
}

QWEN_LANGUAGE_CODES = tuple(code for code in QWEN_LANGUAGE_NAMES if code != "fil")
LANGUAGE_ALIASES = {"fil": "tl"}


def language_code(language) -> Optional[str]:
    code = str(language or "").strip().lower()
    if not code or code == AUTO_LANGUAGE:
        return None
    code = code.split("-")[0].split("_")[0]
    return LANGUAGE_ALIASES.get(code, code)


def language_name(code: Optional[str]) -> str:
    if code is None:
        return "Auto-detect"
    return WHISPER_LANGUAGE_NAMES.get(code, code)


def sorted_by_name(codes) -> list:
    return sorted(codes, key=lambda code: language_name(code).casefold())
