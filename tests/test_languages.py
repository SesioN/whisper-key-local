import logging

from whisper_key.languages import (QWEN_LANGUAGE_CODES, QWEN_LANGUAGE_NAMES, WHISPER_LANGUAGE_NAMES, language_code,
                                   language_name, sorted_by_name)
from whisper_key.model_registry import AUTOMATIC_LANGUAGE, FIXED_LANGUAGE, SELECTABLE_LANGUAGE, ModelRegistry


def test_language_code_normalization():
    assert language_code(" DE-at ") == "de"
    assert language_code("pt_BR") == "pt"
    assert language_code("fil") == "tl"
    assert language_code("Auto") is None and language_code(None) is None and language_code("") is None


def test_names_and_sorting():
    assert language_name("de") == "German" and language_name(None) == "Auto-detect" and language_name("xx") == "xx"
    assert sorted_by_name(["de", "nl", "en"]) == ["nl", "en", "de"]


def test_language_tables_are_consistent():
    assert len(WHISPER_LANGUAGE_NAMES) == 100
    assert set(QWEN_LANGUAGE_CODES) <= set(WHISPER_LANGUAGE_NAMES)
    assert all(code in QWEN_LANGUAGE_NAMES for code in QWEN_LANGUAGE_CODES)


def make_registry():
    return ModelRegistry({
        "tiny": {},
        "small.en": {},
        "german": {"source": "x/german-ct2", "language": "DE"},
        "parakeet": {"engine": "onnx_asr", "onnx_model_type": "nemo-conformer-tdt"},
        "qwen": {"engine": "onnx_asr", "onnx_model_type": "speech-llm"},
        "custom": {"source": "x/custom-ct2", "languages": ["de", "EN", "xx", "de"]},
        "auto-only": {"source": "x/auto-ct2", "languages": []},
    })


def test_language_choice_per_model_family():
    registry = make_registry()
    assert registry.language_choice("tiny") == SELECTABLE_LANGUAGE and len(registry.selectable_languages("tiny")) == 100
    assert registry.language_choice("small.en") == FIXED_LANGUAGE
    assert registry.language_choice("german") == FIXED_LANGUAGE and registry.forced_language("german") == "de"
    assert registry.language_choice("parakeet") == AUTOMATIC_LANGUAGE
    assert registry.selectable_languages("qwen") == list(QWEN_LANGUAGE_CODES)
    assert registry.selectable_languages("custom") == ["de", "en"]
    assert registry.language_choice("auto-only") == AUTOMATIC_LANGUAGE


def test_transcription_language_validates_the_choice(caplog):
    registry = make_registry()
    assert registry.transcription_language("tiny", "fr") == "fr"
    assert registry.transcription_language("small.en", "fr") == "en"
    assert registry.transcription_language("tiny", "auto") is None
    with caplog.at_level(logging.WARNING):
        assert registry.transcription_language("qwen", "sw") is None
        assert registry.transcription_language("qwen", "sw") is None
    assert caplog.text.count("does not support language") == 1
    assert registry.transcription_language("parakeet", "de") is None
