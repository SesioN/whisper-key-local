import logging

from whisper_key.config_manager import ConfigManager, validate_config


def test_defaults_load_with_precheck_thresholds_and_model_languages():
    config = ConfigManager(use_user_settings=False).config
    assert config['vad']['vad_precheck_onset_threshold'] > config['vad']['vad_precheck_offset_threshold']
    assert config['whisper']['models']['distil-large-v3.5']['language'] == 'en'
    assert 'dtype' not in config['audio']


def test_offset_above_onset_resets_both_to_defaults():
    defaults = ConfigManager(use_user_settings=False).config
    import copy
    config = copy.deepcopy(defaults)
    config['vad']['vad_precheck_onset_threshold'] = 0.3
    config['vad']['vad_precheck_offset_threshold'] = 0.6
    validated = validate_config(config, copy.deepcopy(defaults), logging.getLogger("test"))
    assert validated['vad']['vad_precheck_onset_threshold'] == defaults['vad']['vad_precheck_onset_threshold']
    assert validated['vad']['vad_precheck_offset_threshold'] == defaults['vad']['vad_precheck_offset_threshold']


def test_language_settings_are_validated():
    import copy
    defaults = ConfigManager(use_user_settings=False).config
    config = copy.deepcopy(defaults)
    config['whisper']['model_languages'] = {'tiny': 'de', 'base': 5, 'small': ' '}
    config['whisper']['recent_languages'] = ['de', 3, 'fr', 'en', 'nl']
    validated = validate_config(config, copy.deepcopy(defaults), logging.getLogger("test"))
    assert validated['whisper']['model_languages'] == {'tiny': 'de'}
    assert validated['whisper']['recent_languages'] == ['de', 'fr', 'en']
    config['whisper']['model_languages'] = ['de']
    assert validate_config(config, copy.deepcopy(defaults), logging.getLogger("test"))['whisper']['model_languages'] == {}


def test_model_languages_are_saved_as_user_overrides():
    from whisper_key.config_manager import _compute_overrides
    defaults = ConfigManager(use_user_settings=False).config
    import copy
    config = copy.deepcopy(defaults)
    config['whisper']['model_languages'] = {'large-v3-turbo': 'de'}
    config['whisper']['recent_languages'] = ['de']
    overrides = _compute_overrides(config, defaults)
    assert overrides['whisper'] == {'model_languages': {'large-v3-turbo': 'de'}, 'recent_languages': ['de']}
