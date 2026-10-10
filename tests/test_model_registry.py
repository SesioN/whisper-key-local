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
