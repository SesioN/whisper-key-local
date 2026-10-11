from whisper_key.model_registry import ModelRegistry


def make_registry():
    return ModelRegistry({
        'tiny': {'label': 'Tiny'},
        'small.en': {'label': 'Small.En'},
        'distil-small.en': {'source': 'Systran/faster-distil-whisper-small.en'},
        'custom': {'source': 'someone/model-ct2', 'supports_prompt': False},
    })


def test_english_only_detection():
    registry = make_registry()
    assert registry.is_english_only('small.en')
    assert registry.is_english_only('distil-small.en')
    assert not registry.is_english_only('tiny')


def test_hf_repo_resolution():
    registry = make_registry()
    assert registry.get_hf_repo('custom') == 'someone/model-ct2'
    assert registry.get_hf_repo('tiny') == 'Systran/faster-whisper-tiny'


def test_supports_prompt_flag():
    registry = make_registry()
    assert not registry.supports_prompt('custom')
    assert registry.supports_prompt('tiny')


def test_forced_language_from_model_or_english_only_name():
    registry = ModelRegistry({
        'distil': {'source': 'distil-whisper/distil-large-v3.5-ct2', 'language': 'en'},
        'german': {'source': 'x/german-ct2', 'language': 'de'},
        'tiny': {},
        'small.en': {},
        'bad': {'language': 'auto'},
    })
    assert registry.forced_language('distil') == 'en'
    assert registry.transcription_language('german', 'fr') == 'de'
    assert registry.transcription_language('small.en', None) == 'en'
    assert registry.transcription_language('tiny', 'fr') == 'fr'
    assert registry.transcription_language('tiny', None) is None
    assert registry.forced_language('bad') is None
